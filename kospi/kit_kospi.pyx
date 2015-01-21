import numpy as np
cimport numpy as np
import pandas as pd
cimport cython

def liquid_underlying(some_symbols):
    """Takes PANDAS series of various instrument symbols and 
    returns symbol code of most liquid KOSPI future"""
    s = pd.Series(some_symbols.value_counts().index.values)
    return s[(s.str[2:6]=='4101').values].values[0]

def underlying_code_by_two_digit_code(code):
    single_codes = ['1','2','3','4','5','6','7','8','9','A','B','C']
    return str(code[0])+dict(zip(single_codes,np.repeat(['3','6','9','C'],3).tolist()))[code[1]]

def underlyings(some_symbols):
    """Takes PANDAS series of various instrument symbols and 
    returns symbol codes of KOSPI index futures"""
    s = pd.Series(some_symbols.value_counts().index.values)
    return s[s.str[2:6]=='4101']

def expirity_filter(exp_code,u):
    """Takes a KOSPI style expiry code and a list of SERIES of symbols --
    Returns only those symbols that match expiry"""
    return u[u.str[6:8]==exp_code]

def underlying_symbol_by_two_digit_code(some_symbols,code):
    """Takes a PANDAS series of symbols + 2 digit expiry code --
    Returns first KOSPI unlderyling symbol that matches"""
    return expirity_filter(underlying_code_by_two_digit_code(code),underlyings(some_symbols)).values[0]

def strikes_from_symbols(symbol_list):
    """Takes list of symbols, returns their strikes"""
    strikes = pd.Series(symbol_list).str[8:11].astype(float)
    strikes[strikes%5 != 0 ] += .5
    return strikes * 100    

def option_symbols(expiry_code,symbol_list):
    """takes an expiry code (2 digit) plus a LIST/ARRAY of KRX Kospi symbols 
    -- returns a pandas series
    -- said panda series is a list of option issue codes 
    that are both options and of matching expiry"""
    s = pd.Series(symbol_list)
    are_options = np.logical_or(s.str[2:4]=='43',s.str[2:4]=='42')
    matching_code = s.str[6:8]==expiry_code
    return s[np.logical_and(are_options,matching_code)]  

def is_kospi(symbol_list):
    """Returns boolean mask if passed in list of symbols
    matches call/put/fut codes of KOSPI options/futures"""
    s = pd.Series(symbol_list)
    is_call = s.str[2:6]=='4201'
    is_put = s.str[2:6]=='4301'
    is_fut = s.str[2:6]=='4101'
    return np.logical_or(is_call,np.logical_or(is_put,is_fut))      

def types_from_symbols(symbol_list):
    """Takes list of symbols, returns their types"""
    res = np.zeros(len(symbol_list))
    is_call = pd.Series(symbol_list).str[2:4]=='42'
    is_put = pd.Series(symbol_list).str[2:4]=='43'
    res[is_call.values] = 1
    res[is_put.values] = -1
    return res    

def options_expiry_mask(symbol_list,expiry_code,include_futs=True):
    """Takes list of symbols, expiry code, bool to include futs or not
    -- returns boolean mask of symbols that are KOSPI opt/fut"""
    s = pd.Series(symbol_list)
    is_call = s.str[2:6]=='4201'
    is_put = s.str[2:6]=='4301'
    is_fut = s.str[2:6]=='4101'
    is_option = np.logical_or(is_call,is_put)
    is_expiry = s.str[6:8]==expiry_code
    if not include_futs:
        return np.logical_and(is_option,is_expiry)
    return np.logical_and(np.logical_or(is_option,is_fut),is_expiry)    


#first pass one symbol at a time
#data should be a 2d array of LONGS
#bid1,bidsize1,ask1,asksize1,bid2,bidsize2,ask2,asksize2,tradeprice,tradesize
# 0  ,  1      , 2,   3 ,      4,     5 ,   6,     7,         8,       9
@cython.cdivision(True)
@cython.boundscheck(False)
def two_level_fix_a3s(np.ndarray[object,ndim=1] symbols,np.ndarray[long,ndim=1] msg_types,np.ndarray[long, ndim=2] data):
    """
     - Array of symbols(object)
     - Array of message types(longs) 3=A3
     - 2d array of LONGS layed out as
      -- bid1,bidsize1,ask1,asksize1,bid2,bidsize2,ask2,asksize2,tradeprice,tradesize"""
    assert(symbols.shape[0] == data.shape[0])
    assert(data.shape[1] == 10)

    cdef:
        dict last_info = {}
        int a3_count = 0
        int a3_violations = 0
        long tradeprice,tradesize
    for i in range(0,symbols.shape[0]):
        if msg_types[i] == 3:
            a3_count+=1
            if not last_info.has_key(symbols[i]):
                a3_violations+=1
                continue
            if data[i,8] == last_info[symbols[i]][0]: #if the A3 price equals previous bid price
                #SHIFT BIDS
                data[i,] = [last_info[symbols[i]][4],last_info[symbols[i]][5],last_info[symbols[i]][0],0,
                            -1,-1,last_info[symbols[i]][2],last_info[symbols[i]][3],data[i,8],data[i,9]]
            elif data[i,8] == last_info[symbols[i]][2]: #if the A3 price equals previous ask price
                #SHIFT ASKS
                data[i,] = [last_info[symbols[i]][2],0,last_info[symbols[i]][6],last_info[symbols[i]][7],
                            last_info[symbols[i]][0],last_info[symbols[i]][1],-1,-1,data[i,8],data[i,9]]
            else:
                #shit yourself
                a3_violations+=1
                tradeprice = data[i,8]
                tradesize = data[i,9]
                data[i,] = last_info[symbols[i]]
                data[i,8] = tradeprice
                data[i,9] = tradesize
            last_info[symbols[i]] = data[i,]
        else:
            last_info[symbols[i]] = data[i,]
    print 'Total A3 Messages: ', a3_count, ' || Number of A3 Assumption Violations: ', a3_violations
    return data    