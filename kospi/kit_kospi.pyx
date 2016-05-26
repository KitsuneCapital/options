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
    
import numpy as np
import sys
cimport cython
cimport numpy as np
from libc.math cimport exp, sqrt, pow, log, erf, abs, M_PI

ctypedef np.double_t DTYPE_t

cdef long START_TIME = 32400000000
cdef long START_TIME_DST = 28800000000
cdef long END_TIME = 54300000000
cdef long END_TIME_DST = 50700000000
cdef long BUCKETSIZE = 1000000
cdef double EPSILON = .0001

#purely for speed reasons
cdef inline double abz(double a) : return a if a >= 0. else -1 * a

##options ONLY version -- do not use for non kospi
cdef inline double wmid(double bidp, double bids, double askp, double asks):
    if (bids+asks)<=0:
        return np.NaN
    tick_width = .01
    if askp>=3.0:
        tick_width = .05
        
    if (askp - bidp) > tick_width:
        return (bidp+askp)/2
    else:
        return (bidp*asks+askp*bids)/(bids+asks)

#any trade that happens above the previous bid is a 'buy'
cdef inline int trade_dir(double prev_bid,double trade_price):
    if trade_price>prev_bid:
        return 1
    return -1

@cython.cdivision(True)
cdef double std_norm_cdf(double x) nogil:
    return 0.5*(1+erf(x/sqrt(2.0)))

@cython.cdivision(True)
cdef double norm_dist(double x) nogil:
    return (1.0/sqrt(2.0*M_PI))*exp(-0.5*x*x)

@cython.cdivision(True)
def delta(double s, double k, double t, double v,
                 double rf, double cp):
    cdef double d1 = (log(s / k) + (0.5 * pow(v, 2)) * t) / (v * sqrt(t))
    return cp * exp(-rf * t) * std_norm_cdf(cp * d1)

@cython.cdivision(True)
def vega(double s, double k, double t, double v,
                 double rf, double cp):
    cdef double d1 = (log(s / k) + (0.5 * pow(v, 2)) * t) / (v * sqrt(t))
    return s * exp(-rf * t) * sqrt(t) * norm_dist(d1)


@cython.cdivision(True)
@cython.boundscheck(False)
def black_scholes(double s, double k, double t, double v,
                 double rf, double cp):

    cdef double d1, d2, optprice
    with nogil:
        d1 = (log(s / k) + (0.5 * pow(v, 2)) * t) / (v * sqrt(t))
        d2 = d1 - v * sqrt(t)
        optprice = exp(-rf * t) * (cp * s * std_norm_cdf(cp * d1) - cp * k * std_norm_cdf(cp * d2))
    return optprice

@cython.cdivision(True)
@cython.boundscheck(False)
def implied_vol(double underlying, double price, double strike, double t, double rf, double cp):
    cdef long i = 0
    cdef double prices_guess, vol_guess = 1
    cdef double diff, delt
    
    for i in range(0,20):
        price_guess = black_scholes(underlying,strike,t,vol_guess,rf,cp)
        diff = price - price_guess
        if abs(diff) < .001:
            return vol_guess
        vegalol = vega(underlying,strike,t,vol_guess,rf,cp)
        if vega<.01:
            return -1
        vol_guess += diff / vegalol
    return -1

@cython.cdivision(True)
def implied_fut(double guess, double price, double strike, double t, double rf, double sigma, double cp):
    cdef long i
    cdef double prices_guess, underlying_guess = guess
    cdef double diff, delt
    
    if price <= .01:
        return np.NaN
    
    for i in range(20):
        price_guess = black_scholes(underlying_guess,strike,t,sigma,rf,cp)
        diff = price - price_guess
        if abs(diff) < .001:
            return underlying_guess
        delt = delta(underlying_guess,strike,t,sigma,rf,cp)
        underlying_guess += diff / delt
    return np.NaN

def twwmid_cython(np.ndarray[double, ndim=2] md, np.ndarray[long,ndim=1] times, np.ndarray[long, ndim=1] sym_enum,int dst):
    
    cdef long start_time_ref = START_TIME
    cdef long end_time_ref = END_TIME
    
    if dst!=0:
        start_time_ref = START_TIME_DST
        end_time_ref = END_TIME_DST
    
    cdef long md_length = md.shape[0], num_syms = max(sym_enum)+1, num_buckets = (end_time_ref -start_time_ref)/BUCKETSIZE
    cdef long i =0, col=-1, bucket = -1, prev_bucket = -1, micros_past = 0, remainder = 0
    
    cdef np.ndarray[DTYPE_t, ndim=2] wmids = np.zeros([num_buckets,num_syms], dtype=np.double)
    
    cdef np.ndarray[DTYPE_t, ndim=1] last_prices = np.zeros(num_syms, dtype=np.double)
    cdef np.ndarray[DTYPE_t, ndim=1] last_micros = np.zeros(num_syms, dtype=np.double)
    
    
    for i in range(0,md_length):
        if times[i] >= start_time_ref and times[i] < end_time_ref:
            bucket = ((long)(times[i] - start_time_ref)) / BUCKETSIZE
            col = sym_enum[i]
            prev_bucket = ((long)(last_micros[col] - start_time_ref)) / BUCKETSIZE
        
            if bucket >= 0:
                micros_past = ((long)(times[i] - start_time_ref)) % BUCKETSIZE
                if prev_bucket < 0: #FIRST BUCKET OF THE DAY SON
                    wmids[bucket,col] +=  micros_past * wmid(md[i,0],md[i,1],md[i,2],md[i,3])
                else:
                    if (bucket!=prev_bucket): #SWITCHING BUCKETS!!
                        remainder = BUCKETSIZE  - (((long)(last_micros[col] - START_TIME)) % BUCKETSIZE)
                        wmids[prev_bucket][col] += remainder * last_prices[col]
                        wmids[prev_bucket][col] /= BUCKETSIZE
                        prev_bucket+=1
                        while(prev_bucket<bucket):
                            wmids[prev_bucket][col] = last_prices[col]
                            prev_bucket+=1
                        wmids[bucket][col] += micros_past * last_prices[col]
                    else: #OBSERVATION IN SAME BUCKET
                        wmids[bucket][col] += ((long)(times[i] - last_micros[col])) * last_prices[col]
            last_prices[col] = wmid(md[i,0],md[i,1],md[i,2],md[i,3])
            last_micros[col] = (long)(times[i])
    
    #So the above code is CLOSE -- you need to still fix the last bucket in the day
    
    for i in range(0,num_syms):
        
        prev_bucket = ((long)(last_micros[i] - start_time_ref)) / BUCKETSIZE;
        if prev_bucket >=0 : #THIS ENSURES THE SYMBOL TRADED -- YES THIS MATTERS
            remainder = BUCKETSIZE  - (((long)(last_micros[i] - start_time_ref)) % BUCKETSIZE)
            wmids[prev_bucket][i] += remainder * last_prices[i]
            wmids[prev_bucket][i] /= BUCKETSIZE
            #and then 'fill' down
            prev_bucket+=1
            while(prev_bucket<num_buckets):
                wmids[prev_bucket][i] = last_prices[i]
                prev_bucket+=1
    return wmids

def imp_vols_cython(np.ndarray[double, ndim=2] mids, np.ndarray[DTYPE_t,ndim=1] underlyings, np.ndarray[DTYPE_t,ndim=1] strikes, np.ndarray[long,ndim=1] types, double tte, double ir):
    cdef long num_buckets = mids.shape[0],num_syms=mids.shape[1]
    cdef long i =0,j=0
    cdef np.ndarray[DTYPE_t, ndim=2] vols = np.zeros([num_buckets,num_syms], dtype=np.double) * np.NaN
    for i in range(0,num_buckets):
        if underlyings[i]>0 and underlyings[i]!=np.NaN:
            for j in range(0,num_syms):
                if mids[i,j] != np.NaN:
                    if mids[i,j]<=.01: #it's at one bro
                        vols[i,j] = -1
                    else:
                        if (types[j]==-1 and strikes[j]>underlyings[i]) or (types[j]==1 and strikes[j]<underlyings[i]):
                            vols[i,j] = -1
                        else:
                            vols[i,j] = implied_vol(underlyings[i], mids[i,j], strikes[j], tte, ir, types[j])
            for j in range(0,num_syms):
                if vols[i,j]!= np.NaN and vols[i,j] < 0 : #this is placeholder
                    if j<(num_syms/2):#call
                        if strikes[j]!=strikes[(j+num_syms/2)]:
                            print "MAJOR ASSUMPTION ERROR -- CHECK"
                        vols[i,j] = vols[i,(j+num_syms/2)]
                    else:
                        if strikes[j]!=strikes[(j-num_syms/2)]:
                            print "MAJOR ASSUMPTION ERROR -- CHECK"
                        vols[i,j] = vols[i,(j-num_syms/2)]
    return vols

def tick_theos_cython(np.ndarray[double, ndim=2] md, np.ndarray[long,ndim=1] times, np.ndarray[long, ndim=1] sym_enum,
               np.ndarray[double, ndim=2] vols, np.ndarray[long,ndim=1] vol_times,np.ndarray[DTYPE_t,ndim=1] syn_front, np.ndarray[DTYPE_t,ndim=1] syn_back,
               np.ndarray[DTYPE_t,ndim=1] strikes, np.ndarray[long,ndim=1] types, np.ndarray[long,ndim=1] front_back,
               long underlying_sym, double tte_front, double tte_back,double ir):
    
    cdef long md_length = md.shape[0], vol_length = vols.shape[0], num_syms = strikes.shape[0]
    cdef long i =0,j=0,k=0
    cdef double front_underlying,back_underlying, last_bid, last_ask
    cdef np.ndarray[DTYPE_t, ndim=2] theos = np.zeros([md_length,2], dtype=np.double) * np.NaN
    cdef np.ndarray[DTYPE_t, ndim=1] deltas = np.zeros(md_length, dtype=np.double) * np.NaN
    cdef np.ndarray[DTYPE_t, ndim=1] vegas = np.zeros(md_length, dtype=np.double) * np.NaN
    cdef np.ndarray[DTYPE_t, ndim=1] imp_vols = np.zeros(md_length, dtype=np.double) * np.NaN
    cdef np.ndarray[DTYPE_t, ndim=2] imp_futs = np.zeros([md_length,6], dtype=np.double) * np.NaN
    #print md_length,vol_length,num_syms,times.shape[0],vol_times.shape[0],syn_front.shape[0],syn_back.shape[0]
    for i in range(0,md_length):
        #advance to most recent theos observation -- this is when underlying changes
        if (j+1) < vol_length and times[i] >= vol_times[j+1]:
            j+=1
        #update underlying prices
        if sym_enum[i] == underlying_sym and md[i,2]>0:
            front_underlying = wmid(md[i,0],md[i,1],md[i,2],md[i,3]) + syn_front[j]
            back_underlying = wmid(md[i,0],md[i,1],md[i,2],md[i,3]) + syn_back[j]
            vegas[i] = 0
            theos[i,0] = md[i,0]
            theos[i,1] = md[i,2]
            deltas[i] = 1
            imp_vols[i] = np.NaN
            last_bid = md[i,0]
            last_ask = md[i,2]
            for k in range(0,6):
                imp_futs[i,k] = md[i,k]
        elif md[i,2]>0: #okay, fine, you're an option (and it's not offered at zero)
            if front_back[(sym_enum[i]-1)] == 1: #it's a front month option
                if front_underlying != np.NaN and front_underlying>EPSILON:
                    if vols[j,(sym_enum[i]-1)] != np.NaN:
                       theos[i,0] = black_scholes(last_bid+syn_front[j], strikes[(sym_enum[i]-1)], tte_front, vols[j,(sym_enum[i]-1)], ir, types[(sym_enum[i]-1)])
                       theos[i,1] = black_scholes(last_ask+syn_front[j], strikes[(sym_enum[i]-1)], tte_front, vols[j,(sym_enum[i]-1)], ir, types[(sym_enum[i]-1)])
                       deltas[i] = delta(front_underlying, strikes[(sym_enum[i]-1)], tte_front, vols[j,(sym_enum[i]-1)], ir, types[(sym_enum[i]-1)])
                       vegas[i] = vega(front_underlying, strikes[(sym_enum[i]-1)], tte_front, vols[j,(sym_enum[i]-1)], ir, types[(sym_enum[i]-1)])
                       imp_vols[i] = vols[j,(sym_enum[i]-1)]
                       if abz(deltas[i]) > .15: #don't calculate implied fut of low delta crap
                           for k in range(0,6):
                               if k % 2 == 0: #price
                                   imp_futs[i,k] = implied_fut(front_underlying, md[i,k], strikes[(sym_enum[i]-1)], tte_front, ir,vols[j,(sym_enum[i]-1)], types[(sym_enum[i]-1)]) - syn_front[j]
                               else: #size
                                   imp_futs[i,k] = abz(deltas[i]) * md[i,k]
                    else:
                        theos[i,0] = np.NaN
                        theos[i,1] = np.NaN
                        deltas[i] = np.NaN
                        vegas[i] = np.NaN
                        imp_vols[i] = np.NaN
                else:
                    theos[i,0] = np.NaN
                    theos[i,1] = np.NaN
                    deltas[i] = np.NaN
                    vegas[i] = np.NaN
                    imp_vols[i] = np.NaN
            else:
                if back_underlying != np.NaN and back_underlying>EPSILON and vols[j,(sym_enum[i]-1)] != np.NaN:
                    theos[i,0] = black_scholes(last_bid+syn_back[j], strikes[(sym_enum[i]-1)], tte_back, vols[j,(sym_enum[i]-1)], ir, types[(sym_enum[i]-1)])
                    theos[i,1] = black_scholes(last_ask+syn_back[j], strikes[(sym_enum[i]-1)], tte_back, vols[j,(sym_enum[i]-1)], ir, types[(sym_enum[i]-1)])
                    deltas[i] = delta(back_underlying, strikes[(sym_enum[i]-1)], tte_back, vols[j,(sym_enum[i]-1)], ir, types[(sym_enum[i]-1)])
                    vegas[i] = vega(back_underlying, strikes[(sym_enum[i]-1)], tte_back, vols[j,(sym_enum[i]-1)], ir, types[(sym_enum[i]-1)])
                    imp_vols[i] = vols[j,(sym_enum[i]-1)]
                    if abz(deltas[i]) > .15: #don't calculate implied fut of low delta crap
                           for k in range(0,6):
                               if k % 2 == 0: #price
                                   imp_futs[i,k] = implied_fut(back_underlying, md[i,k], strikes[(sym_enum[i]-1)], tte_back, ir,vols[j,(sym_enum[i]-1)], types[(sym_enum[i]-1)]) - syn_back[j]
                               else: #size
                                   imp_futs[i,k] = abz(deltas[i]) * md[i,k]
                else:
                    theos[i,0] = np.NaN
                    theos[i,1] = np.NaN
                    deltas[i] = np.NaN
                    vegas[i] = np.NaN
                    imp_vols[i] = np.NaN
        
    return [theos,deltas,vegas,imp_vols,imp_futs]

def net_effect_cython(np.ndarray[double, ndim=2] md,np.ndarray[long, ndim=1] sym_enum):
    cdef np.ndarray[DTYPE_t, ndim=2] last_info = np.zeros([sym_enum.shape[0],6], dtype=np.double) * np.NaN
    cdef long md_length = md.shape[0], col = -1, i ,j
    cdef np.ndarray[DTYPE_t, ndim=1] effect = np.zeros(md_length, dtype=np.double) * np.NaN
    cdef double bid_change,ask_change 
    for i in range(0,md_length):
        col = sym_enum[i]
        if md[i,4]>EPSILON: #there's a trade
            effect[i] = trade_dir(md[i,0],md[i,4]) * md[i,5] # trade_side * num traded 
        else:
            effect[i] = 0
        for j in range(0,6):
            last_info[col,j] = md[i,j]
    return effect

def net_effect_windowed_cython(np.ndarray[double, ndim=2] effects, np.ndarray[long,ndim=1] times, long window_size):
    cdef long md_length = effects.shape[0], col = -1, i ,j, window_start
    cdef np.ndarray[DTYPE_t, ndim=2] window_effect = np.zeros([md_length,2], dtype=np.double)
    for i in range(0,md_length):
        window_start = times[i] - window_size
        j = i
        while times[j]>= window_start and j>=0:
            window_effect[i,0] += effects[j,0]
            window_effect[i,1] += effects[j,1]
            j-=1
    return window_effect
    
