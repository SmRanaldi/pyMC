# DECIDE WHERE TO PUT IT AND OPEN A GIT REPO


# !pip install PyWavelets

import numpy as np
import pywt
import matplotlib.pyplot as plt
import scipy.signal as sgn
from sklearn.decomposition import NMF

FS_EMG = 1000
FS_MARKER = 250
DS_FACTOR = 1
N_REPLICATES = 10
HAMP_ON = False
FC=2

# ---------- MOVE ELSEWHERE

RESOLUTION = 16
DYNAMIC = 2.4
FEGAIN = 192
FS = 2048
EPOCH = 500
MINAMP = 0
MAXAMP = 1
EMGCHNUM = 32
MAXLEV = 2**RESOLUTION  - 1

# Read Meacs Data
def readMeacsFile(filepathin): # MAKE MORE CHECKS
    with open(filepathin, 'rb') as f:
        data = np.fromfile(f, np.uint16)
        data = np.array(data, dtype=np.float64)
        data *= DYNAMIC/(MAXLEV*FEGAIN)
        data = data.flatten().reshape(-1,EMGCHNUM)
    return data

# -----------------------------

NMF_OPTIONS = {
    'solver': 'cd',
    'max_iter': 5000,
    'tol': 1e-4,
    'init': 'nndsvd'
}


def simple_nmf(data_in, n_components, w=None, h=None, max_iter=1000, tol=1e-4):
    c = 0
    e = np.zeros((max_iter, 1))
    if w is None:
        w = np.random.rand(data_in.shape[0], n_components)
    if h is None:
        h = np.random.rand(n_components, data_in.shape[1])
    while c < max_iter:
        h *= (w.transpose()@data_in)/(w.transpose()@(w@h))
        w *= (data_in@h.transpose())/((w@h)@h.transpose())
        e[c] = np.sqrt(
            np.sum((data_in.flatten() - (w@h).flatten())**2)/data_in.size)
        c += 1
        if c > 10:
            if np.abs(e[c-1] - e[c-10]) < tol:
                break
    if c >= max_iter:
        print(f'No convergence with {n_components} components')
    return w, h


def plot_emg_matrix(data_in):
    fig, ax = plt.subplots(4, 2, figsize=(8, 8))
    for i in range(data_in.shape[1]):
        row = i//2
        col = i % 2
        ax[row, col].plot(data_in[:, i])
    plt.show()

# Hampel filter


def hampel_EMG(data_in, w=100, sigma=3):
    data_out = data_in.flatten()
    for i in range(data_out.size):
        first_point = np.max([0, i-w])
        last_point = np.min([data_out.size, i+w])
        segm = data_in[first_point:last_point]
        dev = sigma*np.std(segm)
        med = np.median(
            data_in[np.max([0, i-3*w]):np.min([data_in.size, i+3*w])])
        if np.abs(med-data_out[i]) > dev:
            data_out[i] = med
    return data_out

# Filter EMG


def filter_EMG(data_in):
    b, a = sgn.butter(4, np.array([20, 450])/(FS_EMG/2), btype='bandpass')
    bb, aa = sgn.butter(4, np.array([49.5, 50.5])/(FS_EMG/2), btype='bandstop')
    signal_tmp = sgn.filtfilt(b, a, data_in, axis=0)
    data_hamp = sgn.filtfilt(bb, aa, signal_tmp, axis=0)
    signal_out = data_hamp.copy()
    if HAMP_ON:
        for i in range(data_in.shape[1]):
            signal_out[:, i] = hampel_EMG(data_hamp[:, i])

    return signal_out

# Classical envelope extraction


def envelope_EMG(data_in, fc=FC):
    b, a = sgn.butter(4, fc/(FS_EMG/2), btype='lowpass')
    env = sgn.filtfilt(b, a, np.abs(filter_EMG(data_in)), axis=0)
    env[np.where(env < 1e-4*np.max(env))] = 1e-4*np.max(env)

    return env

# --- Normalization via events


def normalize_EMG(data_in, events_in):
    norm_factor = []
    for evt in zip(events_in[:-1], events_in[1:]):
        e1 = int(evt[0])
        e2 = int(evt[1])
        norm_factor.append(np.max(data_in[e1:e2, :], axis=0))
    norm_factor = np.asarray(norm_factor)
    norm_factor = np.median(norm_factor, axis=0)
    signal_out = data_in
    for i in range(signal_out.shape[1]):
        signal_out[:, i] /= norm_factor[i]
    signal_out[signal_out<1e-4] = 1e-4
    return signal_out


# --- VAF Function
def VAF(true_data, rec_data):

    return 1 - np.sum((true_data.flatten() - rec_data.flatten())**2)/np.sum(true_data.flatten()**2)
    # return 1 - np.sum((true_data.flatten() - rec_data.flatten())**2)/(np.sum((rec_data.flatten() - np.mean(rec_data.flatten()))**2))


def DoFWavelets(signal_in, events_in, ds_factor=DS_FACTOR):
    out = []
    wav = 'db5'
    events = [int(x*ds_factor) for x in events_in]
    for evt in zip(events[:-1], events[1:]):
        signal = signal_in[evt[0]:evt[1]]
        signal -= np.mean(signal)
        n = int(np.floor(np.log2(signal.size)))
        ca, cd = pywt.dwt(signal, wav)
        lca = [ca.size]
        rca = sgn.resample(ca, signal.size)
        rcd = sgn.resample(cd, signal.size)
        rca -= np.mean(rca)
        rcd -= np.mean(rcd)
        pca = []
        pcd = []
        pca.append(sgn.periodogram(rca, nfft=rca.size)[1])
        pcd.append(sgn.periodogram(rcd, nfft=rcd.size)[1])
        for i in range(n-2):
            ca, cd = pywt.dwt(ca, wav)
            lca.append(ca.size)
            rca = sgn.resample(ca, signal.size)
            rcd = sgn.resample(cd, signal.size)
            rca -= np.mean(rca)
            rcd -= np.mean(rcd)
            pca.append(sgn.periodogram(rca, nfft=rca.size)[1])
            pcd.append(sgn.periodogram(rcd, nfft=rcd.size)[1])
        ps = sgn.periodogram(signal, nfft=signal.size)[1]
        cpCD = []
        for i in range(n-1):
            L = int(np.round(ps.size/(2**i)))
            cpCD.append(np.corrcoef(
                pcd[i][0:L].flatten()/(np.linalg.norm(pcd[i][0:L])+1e-7),
                ps[0:L].flatten()/(np.linalg.norm(ps[0:L])+1e-7)
            )[1, 0])
        cpCD = np.asarray(cpCD)
        pos = np.where(np.diff(cpCD) > -0.05*np.mean(np.abs(np.diff(cpCD))))[0]
        if np.any(pos):
            idxCD = np.min(pos)
            if idxCD > len(lca) - 2:
                idxCD = len(lca) - 2
        else:
            idxCD = len(lca)-2
        tmp_value = lca[idxCD+1] * \
            (1 + np.min([1, np.sum(pcd[idxCD+1])/np.sum(pca[idxCD+1])]))
        # tmp_value = lca[idxCD+1]
        if np.isnan(tmp_value):
            tmp_value = 1
        out.append(tmp_value)
    return out


def NoiseHVar(rec_in):
    nH = np.zeros(rec_in.shape[1])
    for i in range(rec_in.shape[1]):
        nH[i] = np.std(rec_in[:, i].flatten())
    out = np.ones(rec_in.shape)
    for i in range(rec_in.shape[1]):
        out[:, i] *= nH[i]
    return out


def likelihood(W_in, H_in, M):
    rec = (W_in@H_in).transpose()
    noise = NoiseHVar(rec)
    L = []
    for i in range(M.shape[1]):
        L.append(np.sum((rec[:, i].flatten() - M[:, i].flatten())
                 ** 2 / (np.var(M.flatten()) + noise[:, i].flatten())))
    return np.sum(np.asarray(L))


def AIC(W_in, H_in, M, events_in, ds_factor=DS_FACTOR):
    dof = 0
    for i in range(H_in.shape[0]):
        dof += np.sum(np.asarray(DoFWavelets(
            H_in[i, :], events_in, ds_factor)))
    L = likelihood(W_in, H_in, M)
    out = L + 2*W_in.shape[1]*W_in.shape[0] + 2*dof
    return out

# --- Remove negative values


def remove_negative(data_in):
    data_out = data_in.copy()
    for i in range(data_in.shape[1]):
        ths = 1e-6*np.max(data_in[:, i])
        data_out[data_in[:, i] < ths, i] = ths
    return data_out

# --- Downsample EMG


def downsample_EMG(data_in, ds_factor):
    # data_out = data_in[::ds_factor,:]
    if ds_factor > 1:
        data_out = sgn.resample(data_in, int(
            data_in.shape[0]/ds_factor), axis=0)
    else:
        data_out = data_in.copy()
    return data_out

# --- Prepare EMG for synergy analysis


def prepare_EMG(emg_in, events_in, ds_factor=DS_FACTOR):
    emg_cut, evt = cut_EMG(emg_in, events_in)
    emg_filt = filter_EMG(emg_cut)
    env = envelope_EMG(data_in=emg_filt)
    if env.shape[0] < env.shape[1]:
        env = env.transpose()
    if events_in is not None:
        env_norm = normalize_EMG(data_in=env, events_in=events_in)
    env_down = downsample_EMG(env_norm, ds_factor)
    out = remove_negative(env_down)
    return out, evt


def cut_EMG(data_in, events_in):
    data_out = data_in[int(events_in[0]):int(events_in[-1]), :]
    events_out = events_in - events_in[0]
    return data_out, events_out

# --- Synergy extractor


def extract_synergies(env, nsyn):

    # env, evt= prepare_EMG(emg_in=emg_in, events_in=events_in, ds_factor=ds_factor)

    nmf = NMF(n_components=nsyn, **NMF_OPTIONS)
    H = nmf.fit_transform(env).transpose()
    W = nmf.components_.transpose()
    for j in range(nsyn):
        H[j, :] *= np.linalg.norm(W[:, j])
        W[:, j] /= np.linalg.norm(W[:, j])
    rec = (W@H).transpose()
    vaf = VAF(env, rec)

    return W, H, vaf


# --- Nsyn selection with VAF
def nsyn_selection_VAF(env, max_n=None, n_replicates=1):

    n_muscles = env.shape[1]
    if max_n is None:
        max_n = n_muscles

    VAF_curve_all = np.zeros((n_replicates, max_n))
    VAF_muscles_all = np.zeros((n_replicates, max_n, n_muscles))

    for k in range(n_replicates):
        VAF_curve = []
        VAF_muscles = []
        for i in range(max_n):
            nmf = NMF(n_components=i+1, **NMF_OPTIONS)
            H = nmf.fit_transform(env).transpose()
            W = nmf.components_.transpose()
            rec = (W@H).transpose()
            VAF_curve_all[k, i] = VAF(env, rec)
            for j in range(n_muscles):
                VAF_muscles_all[k, i, j] = VAF(env[:, j], rec[:, j])

    VAF_curve = np.mean(VAF_curve_all, axis=0).squeeze()
    VAF_muscles = np.mean(VAF_muscles_all, axis=0).squeeze()

    return np.asarray(VAF_curve), np.asarray(VAF_muscles)

# --- Nsyn selection curves


def nsyn_selection(env, evt, max_n=None, ds_factor=DS_FACTOR, n_replicates=1):

    # env, evt = prepare_EMG(emg_in=emg_in, events_in=events_in, ds_factor=ds_factor)
    n_muscles = env.shape[1]
    if max_n is None:
        max_n = n_muscles

    VAF_curve_all = np.zeros((n_replicates, max_n))
    aic_all = np.zeros((n_replicates, max_n))
    VAF_muscles_all = np.zeros((n_replicates, max_n, n_muscles))

    for k in range(n_replicates):
        VAF_curve = []
        VAF_muscles = []
        aic = []
        for i in range(max_n):
            nmf = NMF(n_components=i+1, **NMF_OPTIONS)
            H = nmf.fit_transform(env).transpose()
            W = nmf.components_.transpose()
            rec = (W@H).transpose()
            VAF_curve.append(VAF(env, rec))
            VAF_muscles.append([VAF(env[:, j], rec[:, j])
                               for j in range(n_muscles)])
            aic.append(AIC(W, H, env, evt, ds_factor))
            VAF_curve_all[k, i] = VAF(env, rec)
            aic_all[k, i] = AIC(W, H, env, evt, ds_factor)
            for j in range(n_muscles):
                VAF_muscles_all[k, i, j] = VAF(env[:, j], rec[:, j])

    VAF_curve = np.mean(VAF_curve_all, axis=0).squeeze()
    VAF_muscles = np.mean(VAF_muscles_all, axis=0).squeeze()
    aic = np.mean(aic_all, axis=0).squeeze()

    return np.asarray(VAF_curve), np.asarray(VAF_muscles), np.asarray(aic)


# --- Correlation between W
def W_dot(W1, W2):
    return np.dot(W1/np.linalg.norm(W1), W2/np.linalg.norm(W2))


# --- Sort via a template
# Data needs to be [N_muscles x N_synergies]. Size W_in >= size W_temp
def sort_W(W_temp, W_in):
    n_syn_common = np.min([W_temp.shape[1], W_in.shape[1]])
    all_d = []
    for i in range(W_temp.shape[1]):
        all_d_tmp = []
        for j in range(W_in.shape[1]):
            all_d_tmp.append(W_dot(W_temp[:, i], W_in[:, j]))
        all_d.append(all_d_tmp)
    for i in range(W_in.shape[1] - n_syn_common):
        mm = np.min(np.vstack(all_d).flatten())
        all_d.append(0.2*mm*np.ones((1, W_in.shape[1])))
    all_d = np.vstack(all_d)  # Rows: reference. Columns: data to sort
    idx_out = np.zeros(W_in.shape[1], dtype=int)
    for i in range(W_in.shape[1]):
        max_ref = np.unravel_index(all_d.argmax(), all_d.shape)
        idx_out[max_ref[0]] = max_ref[1]
        all_d[max_ref[0], :] = 0
        all_d[:, max_ref[1]] = 0
    return idx_out


# --- Non-negative reconstruction
def nnr(data_in, w_in, ds_factor=DS_FACTOR):

    # data_in, evt = prepare_EMG(emg_in=emg_in, events_in=events_in, ds_factor=ds_factor)
    max_iter = NMF_OPTIONS['max_iter']
    tol = NMF_OPTIONS['tol']
    if data_in.shape[1] < data_in.shape[0]:
        data_in = data_in.transpose()
    if w_in.shape[0] < w_in.shape[1]:
        w_in = w_in.transpose()
    c = 0
    convergence = False
    err = []
    h = np.random.rand(w_in.shape[1], data_in.shape[1])
    while c < max_iter:
        num = w_in.transpose()@data_in
        den = w_in.transpose()@w_in@h
        h *= num/den
        err.append(np.sqrt(np.sum((data_in.flatten() - (w_in@h).flatten())**2)))

        if c > 10:
            if np.abs(err[-10] - err[-1]) < tol:
                c = max_iter
                convergence = True
        c += 1

    if not convergence:
        print('Reconstruction algorithm did not converge')

    rec = (w_in@h)
    vaf_ = VAF(data_in, rec)

    return h, vaf_


# Visualize synergies
def plot_synergies(W_in, H_in, labels=None, labels_bar=None):
    width_bar = 0.8
    if W_in.ndim < 3:
        W_plt = W_in.reshape(-1, W_in.shape[0], W_in.shape[1])
    else:
        W_plt = W_in
    if labels_bar is None:
        labels_bar = [None for i in range(W_plt.shape[0])]
    all_widths = np.linspace(-W_plt.shape[0]/2,
                             W_plt.shape[0]/2, W_plt.shape[0], dtype='int')
    n_syn = W_plt.shape[2]
    fig, ax = plt.subplots(n_syn, 2, figsize=(20, 10))
    ax = ax.reshape(-1,2)
    for i in range(n_syn):
        for k in range(W_plt.shape[0]):
            ax[i, 0].bar(np.arange(W_plt.shape[1]) + width_bar*all_widths[k], W_plt[k, :, i].squeeze(),
                         tick_label=labels, label=labels_bar[k], width=width_bar, facecolor=[0.2, 0.2, 0.6])
        ax[i, 1].plot(H_in[i, :], c='r', lw=3)
        ax[i, 0].set_frame_on(False)
        ax[i, 1].set_frame_on(False)
        ax[i, 0].grid(False)
        ax[i, 1].grid(False)
        ax[i, 0].set_yticks([])
        ax[i, 1].set_yticks([])
        ax[i, 0].tick_params(axis='both', length=0)
        ax[i, 1].tick_params(axis='both', length=0)
        if i == n_syn-1:
            if labels is not None:
                ax[i, 0].set_xticklabels(labels, fontsize=18)
            ax[i, 0].tick_params(axis='x', rotation=45, labelsize=22)
            # ax[i,1].tick_params(axis='x', rotation=45, labelsize=14)
            ax[i, 1].set_xticklabels([])
        else:
            ax[i, 0].set_xticklabels([])
            ax[i, 1].set_xticklabels([])
    if labels_bar[0] is not None:
        ax[0, 0].legend(fontsize=12)
    plt.show()
    return fig


def get_random_threshold(data_in, w_in, n_replicates=50, perc=95):
    w_rand = w_in.copy()

    vaf_rand = []
    for i in range(n_replicates):
        np.random.shuffle(w_rand)
        _, vv = nnr(data_in, w_rand)
        vaf_rand.append(vv)
    vaf_rand = np.asarray(vaf_rand)

    return np.percentile(vaf_rand, perc)
