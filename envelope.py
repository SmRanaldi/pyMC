import ctypes
import numpy as np
import pandas as pd
import pycorrelate as pyc
from scipy.stats import chi2
import scipy.signal as sgn
from joblib import Parallel, delayed
from statsmodels.regression.linear_model import yule_walker

import os

import warnings
warnings.filterwarnings("error")

base_path = os.path.join(os.path.dirname(
    os.path.realpath(__file__)))

L_INIT = 200
HAMP_ON = False
FS_EMG = 1000
ALPHA=1
NU=2
MAX_ITER=100
MAX_ORDER=13
THS_CONV=1e-5

P = 0.797884560802866
F = 0.535398163397448 
C = 4*F


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


def adaptive_envelope(data_in):
    chiTable = _load_chi()
    # data = _preWhiten(data_in)
    data = data_in
    if not _white_test(data):
        print("Signal is not white. Results might be inaccurate.")

    # --- Initialization
    m = np.ones(data.shape)*L_INIT
    w = np.zeros(data.shape)
    d1 = np.zeros(data.shape)
    d2 = np.zeros(data.shape)
    nsamples = m.shape[0]

    # --- Static estimation
    for i in range(nsamples):
        tmp = np.abs(data_in[
            np.max([0, i-int(0.5*m[i])]):
            np.min([i+int(0.5*m[i]),nsamples])
        ])
        t=_get_t(tmp.shape[0])
        w[i] = (1/tmp.shape[0])*np.sum(np.abs(tmp)) # THIS IS THE CONVENTIONAL MAV
        d1[i] = np.sum(t * (tmp))
        d2[i] = np.sum((t**2)*(tmp))
        d1[i] /= (np.sum(t**2)*P)
        d2[i] /= (np.sum(t**4)*P)

    m=np.asarray([_update_m(w[i], d1[i], d2[i]) for i in range(nsamples)])
    w=np.asarray([_update_w(data, m[i], i) for i in range(nsamples)])
    # for i in range(nsamples):
    #     m[i] = _single_sample(data, m[i], w[i], d1[i], d2[i], i, chiTable)
    # w=np.asarray([_update_w(data, m[i], i) for i in range(nsamples)])
    return w,m

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
    try:
        t=_get_t(tmp.shape[0])
        a1 = np.sum(t**2)
        a2 = np.sum(t**4)
        d1 = np.sum(t*tmp)/(a1*P)
        # d2 = np.sum((t**2)*(tmp**(1/ALPHA)))/(a2*P) - (a1/(a2*P))*(np.sum((1-(t**2)*(a1/a2))*(tmp**(1/ALPHA))))/(tmp.shape[0]-(a1/a2)*a1)
        d2 = np.sum((t**2)*tmp)/(a2*P) - (a1/(a2*P))*(np.sum((1-(t**2)*(a1/a2))*(tmp)))/(1 + 2*tmp.shape[0]+(a1**2/a2))
        d2*=2
    except RuntimeWarning:
        print(t)
    return d1, d2

def _get_t(l):
    t = np.arange(
        start = -np.floor(l/2),
        stop = np.ceil(l/2),
        step=1
    )
    return t

def _white_test(data_in, alpha=0.05):
    # https://dsp.stackexchange.com/questions/7678/determining-the-whiteness-of-noise
    N = data_in.shape[0]
    data = data_in - np.mean(data_in)
    maxlag = N-1

    cc = pyc.ucorrelate(data, data, maxlag=maxlag) + 1e-6
    R = (N/cc[int(len(cc)/2)] ** 2)*np.sum(cc[int(len(cc)/2):] ** 2)
    # pv = 1 - chi2.cdf(R, maxlag)
    pa = chi2.ppf(1-alpha, maxlag)

    return R > pa

def _avg_win(segment):
    return np.sum(np.abs(segment))/segment.shape[0]

def _load_chi():
    chi = pd.read_csv(os.path.join(
        base_path, 'tables', 'chitable.csv'),
        header=None, dtype=np.double, engine='c').values
    return chi

def _get_white_sample(segment_in, order):
    coeff, _ = yule_walker(segment_in, order)
    return sgn.lfilter(-coeff, 1, segment_in)[int(segment_in.shape[0]/2)]

def _get_start_stop(lin, hl):
    idx = []
    for i in range(lin):
        idx.append((np.max([0,i-hl]), np.min([i+hl,lin])))
    return idx