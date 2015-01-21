import numpy as np
cimport numpy as np
import pandas as pd
cimport cython

def seconds_to_days(seconds_to_expiry):
    """Takes an array/series of times to expiration in SECONDS
    and returns the rounded (up) # of days """
    return np.ceil(seconds_to_expiry /24./ 3600.)

def get_dte(expiry,trade_date):
    """Naive days to expiry calculator -- Business Days but NO holiday magic"""
    drange = pd.date_range(end=pd.Timestamp(expiry),start=pd.Timestamp(trade_date))
    s = (pd.Series(pd.to_datetime(drange).tolist(),index=drange).asfreq(pd.tseries.offsets.BDay()))
    return len(s)

def to_sql_time(a_pandas_timestamp):
    return pd.Timestamp(a_pandas_timestamp).strftime('%Y-%m-%d %H:%M:%S')

def to_sql_date(a_pandas_timestamp):
    return pd.Timestamp(a_pandas_timestamp).strftime('%Y-%m-%d').replace('-','')    

@cython.cdivision(True)
@cython.boundscheck(False)
def fix_timestamps(np.ndarray[long, ndim=1] possibly_duplicate_times):
    """Takes a SORTED array of times in RAW-FROM-EPOCH long form
    and wherever it runs into consecutive duplicates, adds 1 second
    to the time until no duplicates remain"""
    cdef:
        long t_len = possibly_duplicate_times.shape[0]
        long i =0, current_time
        dict time_dict = dict()
        np.ndarray[long, ndim=1] res = np.zeros(t_len, dtype=np.int64)
    for i from 0<= i < t_len:
        current_time = possibly_duplicate_times[i]
        if time_dict.has_key(current_time): #omg its already in here
            res[i] = current_time + time_dict[current_time]
            time_dict[current_time]+=1
        else:
            res[i] = current_time
            time_dict[current_time] = 1
    return res