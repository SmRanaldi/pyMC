import ctypes
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.stats import chi2
import scipy.signal as sgn
from joblib import Parallel, delayed
from statsmodels.regression.linear_model import yule_walker
import cython

cimport numpy as cnp

import os

import warnings
warnings.filterwarnings("error")

cnp.import_array()

base_path = os.path.join(os.path.dirname(
    os.path.realpath(__file__)))

cdef int L_INIT = 200
cdef int HAMP_ON = 0
cdef int FS_EMG = 1000
cdef int ALPHA=1
cdef int NU=2
cdef int MAX_ITER=100
cdef int MAX_ORDER=13
cdef double THS_CONV=1e-5

cdef double P = 0.797884560802866
cdef double F = 0.535398163397448 
cdef double C = 4*F

cdef cnp.ndarray CHI = np.genfromtxt(os.path.join(
        base_path, 'tables', 'chitable.csv'
    ), delimiter=',')

DTYPE = np.float64
ctypedef cnp.int64_t DTYPE_t

def adaptive_envelope(cnp.ndarray data_in):
    cdef cnp.ndarray data_c = np.array(data_in, dtype=DTYPE)
    return _adaptive_envelope_main(data_c)

def _adaptive_envelope_main(cnp.ndarray data_in):

    # --- VARIABLES
    cdef int l_data = data_in.size
    cdef cnp.ndarray data = data_in
    cdef cnp.ndarray m = np.ones((l_data), dtype=DTYPE)*L_INIT
    cdef cnp.ndarray w = np.zeros((l_data), dtype=DTYPE)
    cdef cnp.ndarray d1 = np.zeros((l_data), dtype=DTYPE)
    cdef cnp.ndarray d2 = np.zeros((l_data), dtype=DTYPE)
    cdef int nsamples = m.shape[0]
    cdef cnp.ndarray tmp = np.zeros((1000))
    cdef cnp.ndarray t = np.zeros((1000))
    cdef int i, i_from, i_to, l_tmp
    cdef cnp.ndarray chiTable = CHI

    # data = _preWhiten(data_in)

    # --- Static estimation
    for i in range(nsamples):
        i_from = np.max([0, i-int(0.5*m[i])])
        i_to = np.min([i+int(0.5*m[i]), nsamples])
        l_tmp = i_to-i_from
        tmp[:l_tmp] = np.abs(data[i_from:i_to])
        t[:l_tmp]=_get_t(l_tmp)
        w[i] = (1/l_tmp)*np.sum(np.abs(tmp[:l_tmp])) # THIS IS THE CONVENTIONAL MAV
        d1[i] = np.sum(t[:l_tmp] * (tmp[:l_tmp]))
        d2[i] = np.sum((t[:l_tmp]**2)*(tmp[:l_tmp]))
        d1[i] /= (np.sum(t[:l_tmp]**2)*P)
        d2[i] /= (np.sum(t[:l_tmp]**4)*P)

    for i in range(nsamples):
        m[i] = _single_sample(data, m[i], w[i], d1[i], d2[i], i, chiTable)
    w=np.asarray([_update_w(data, m[i], i) for i in range(nsamples)])
    return w,m

# --------- OLD STUFF --------

# Hampel filter
def hampel_EMG(data_in, w=100, sigma=3):
    data_out = data_in.flatten()
    for i in range(data_out.size):
        first_point = np.max([0, i-w])
        last_point = np.min([data_out.size, i+w])
        segm = data_in[first_point:last_point]
        dev = sigma*np.std(segm)
        med = np.median(data_in[np.max([0, i-3*w]):np.min([data_in.size, i+3*w])])
        if np.abs(med-data_out[i]) > dev:
            data_out[i] = med
    return data_out

# Filter EMG
def filter_EMG(data_in, fs=FS_EMG, hamp_on=HAMP_ON, fl=25, fh=450, fnotch=50):
    np.nan_to_num(data_in, copy=False, nan=1e-6)
    b, a = sgn.butter(4, np.array([fl, fh])/(fs/2), btype='bandpass')
    bb, aa = sgn.butter(4, np.array([fnotch-0.5, fnotch+0.5])/(fs/2), btype='bandstop')
    signal_tmp = sgn.filtfilt(b, a, data_in, axis=0)
    data_hamp = sgn.filtfilt(bb, aa, signal_tmp, axis=0)
    signal_out = data_hamp.copy()
    if hamp_on:
        signal_out = hampel_EMG(data_hamp)

    return signal_out

def mav_envelope_par(data_in, l_win):
    half_l = int(np.floor(0.5*l_win))
    env = np.zeros(data_in.shape)
    idx = _get_start_stop(data_in.shape[0], half_l)
    env = Parallel(n_jobs=-1)((delayed)(_avg_win)(data_in[a[0]:a[1]]) for a in idx)
    return env

def mav_envelope(data_in, l_win):
    half_l = int(np.floor(0.5*l_win))
    env = np.zeros(data_in.shape)
    idx=_get_start_stop(data_in.shape[0], half_l)
    for i, e in enumerate(idx):
        env[i] = np.sum(data_in[e[0]:e[1]])/(e[1]-e[0])
    return env

# ----- AUXILIARY FUNCTIONS ------

def _preWhiten(data_in, wlen=150, order=13):
    if order>MAX_ORDER:
        order = MAX_ORDER
    hl = int(np.floor(wlen/2))
    idx = _get_start_stop(data_in.shape[0], hl)
    data_out = np.zeros(data_in.shape)
    for i,e in enumerate(idx):
        data_out[i] = _get_white_sample(data_in[e[0]:e[1]], order)
    return data_out

def _preWhiten_par(data_in, wlen=150, order=13):
    if order>MAX_ORDER:
        order = MAX_ORDER
    hl = int(np.floor(wlen/2))
    idx = _get_start_stop(data_in.shape[0], hl)
    data_out = np.zeros(data_in.shape)
    with Pool(4) as pool:
        data_out = np.asarray(pool.starmap(_get_white_sample, [(data_in[e[0]:e[1]], order) for e in idx]))
    return data_out

def _est_entropy(l, chiTable):
    # hl = int(np.ceil(l/2))
    ns = chiTable[int(l-1),:]
    ns[ns<1e-6] = 1
    aa = ns * np.log(1/ns)
    ent=np.sum(aa)
    return ent

def _single_sample(signal, m, w, d1, d2, idx, chiTable):
    iteration = 1
    ent_oold = _est_entropy(m, chiTable)
    ent_old = _est_entropy(m, chiTable)
    while iteration<MAX_ITER:
        m_new = _update_m(w, d1, d2)
        w_new = _update_w(signal, m_new, idx)
        d1_new, d2_new = _update_der(signal, m_new, idx)
        ent_new = _est_entropy(m_new, chiTable)
        iteration += 1
        if iteration>2:
            diff1 = ent_new-ent_old
            diff2 = ent_old-ent_oold
            if diff2-diff1<THS_CONV:
                break
        m = m_new
        w, d1, d2 = w_new, d1_new, d2_new
        ent_oold = ent_old
        ent_old = ent_new
    return m

def _update_m(w, d1, d2):
    aa = d1/2
    bb = (1/6)*(d2 + (aa**2)/w)
    n = C*(w**4)
    # d = ((bb*w)+(ALPHA*NU-1)*(aa**2))/2
    d = (bb + 0.5*(aa**2)/(w))**2
    # d = d**2
    m = np.round(np.abs(n/d)**(1/5))
    if np.isnan(m): m=L_INIT
    if m>10000: m = 10000
    if m<2: m=2
    return m

def _update_w(signal, m, idx):
    hl = int(np.floor(m/2))
    tmp = signal[
        np.max([0, idx-hl]):
        np.min([idx+hl, signal.shape[0]])
    ]
    out = np.sum(np.abs(tmp)**2)/tmp.shape[0]
    out = np.sqrt(out)
    return out

def _update_der(signal, m, idx):
    hl = int(np.floor(m/2))
    tmp = np.abs(signal[
        np.max([0, idx-hl]):
        np.min([idx+hl, signal.shape[0]])
    ])
    t=_get_t(tmp.shape[0])
    if tmp.shape[0]>1:
        t=_get_t(tmp.shape[0])
    else:
        t = np.asarray([1])
    a1 = np.sum(t**2)
    a2 = np.sum(t**4)
    d1 = np.sum(t*tmp)/(a1*P)
    # d2 = np.sum((t**2)*(tmp**(1/ALPHA)))/(a2*P) - (a1/(a2*P))*(np.sum((1-(t**2)*(a1/a2))*(tmp**(1/ALPHA))))/(tmp.shape[0]-(a1/a2)*a1)
    d2 = np.sum((t**2)*tmp)/(a2*P) - (a1/(a2*P))*(np.sum((1-(t**2)*(a1/a2))*(tmp)))/(1 + 2*tmp.shape[0]+(a1**2/a2))
    d2*=2
    return d1, d2

def _get_t(l):
    t = np.arange(
        start = -np.floor(l/2),
        stop = np.ceil(l/2),
        step=1
    )
    return t

def _avg_win(segment):
    return np.sum(np.abs(segment))/segment.shape[0]

def _load_chi():
    return CHI

def _get_white_sample(segment_in, order):
    coeff, _ = yule_walker(segment_in, order)
    return sgn.lfilter(-coeff, 1, segment_in)[int(segment_in.shape[0]/2)]

def _get_start_stop(lin, hl):
    idx = []
    for i in range(lin):
        idx.append((np.max([0,i-hl]), np.min([i+hl,lin])))
    return idx