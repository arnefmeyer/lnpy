#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: overflowcheck=False
#cython: nonecheck=False

import numpy as np
cimport numpy as np

np.import_array()

ctypedef np.double_t DOUBLE
ctypedef np.int8_t BYTE
ctypedef np.int64_t LONG
                        

def calc_auc(np.ndarray[DOUBLE, ndim=1, mode='c'] Y,
             np.ndarray[DOUBLE, ndim=1, mode='c'] score,
             int n_steps):
    """
        Wrapper for C-based AUC calculation
    """
    
    cdef int n_samples = Y.shape[0]

    cdef double auc
    cdef np.ndarray[DOUBLE, ndim=1, mode='c'] tpr = \
        np.zeros((n_steps,), np.double, order='c')
    cdef np.ndarray[DOUBLE, ndim=1, mode='c'] fpr = \
        np.zeros((n_steps,), np.double, order='c')
    cdef int spike_cnt = 0
        
    cdef double smin
    cdef double smax
    cdef int i
    cdef int t
    cdef int b

    smin = np.min(score)
    smax = np.max(score)
    spike_cnt = np.sum(Y > 0)

    cdef double balance
    cdef double threshold
    cdef double pred
    cdef double srange
    cdef double delta
    if int(spike_cnt) == n_samples or spike_cnt == 0:
        auc = 0.5

    else:
        srange = np.abs(smax - smin)
        smin -= 1e-3 * srange;
        smax += 1e-3 * srange;
        delta = (srange + 2e-3*srange) / float(n_steps)

        balance = float(spike_cnt) / float(n_samples)
        for b in range(n_steps):
            threshold = smin + b * delta;

            for i in range(n_samples):
                pred = score[i] + threshold;

                if ((Y[i] > 0) and (pred > 0)):
                    tpr[b] = tpr[b] + 1.

                if ((Y[i] <= 0) and (pred > 0)):
                    fpr[b] = fpr[b] + 1

            tpr[b] /= (float(n_samples) * balance)
            fpr[b] /= (float(n_samples) * (1. - balance))

        auc = 0.
        for b in range(n_steps-1):
            auc += (tpr[b] + tpr[b+1]) / 2. * (fpr[b+1] - fpr[b])
        
    return auc
    
    
    
def calc_auc_trials(np.ndarray[DOUBLE, ndim=2, mode='c'] Y,
                    np.ndarray[DOUBLE, ndim=1, mode='c'] score,
                    int n_steps):
    """
        Wrapper for c-based AUC calculation (multiple trials)
    """

    cdef int n_samples = Y.shape[0]
    cdef int n_trials = Y.shape[1]

    cdef np.ndarray[DOUBLE, ndim=1, mode='c'] auc_trials = \
        np.zeros((n_trials,), np.float64, order='c')
    cdef np.ndarray[DOUBLE, ndim=1, mode='c'] tpr = \
        np.zeros((n_steps,), np.float64, order='c')
    cdef np.ndarray[DOUBLE, ndim=1, mode='c'] fpr = \
        np.zeros((n_steps,), np.float64, order='c')
    cdef np.ndarray[DOUBLE, ndim=1, mode='c'] spike_cnt = \
        np.zeros((n_trials,), np.float64, order='c')
        
    cdef double smin = score[0]
    cdef double smax = score[0]
    cdef int i
    cdef int t
    cdef int b
    cdef double spike_cnt_all = 0
    for i in range(n_samples):
        if (score[i] < smin):
            smin = score[i]

        if (score[i] > smax):
            smax = score[i]

        for t in range(n_trials):
            if (Y[i,t] > 0):
                spike_cnt[t] += 1
                spike_cnt_all += 1

    cdef double srange
    cdef double delta
    cdef double balance
    cdef double threshold
    cdef double pred
    cdef double auc
    if int(spike_cnt_all) == n_samples * n_trials or spike_cnt_all == 0:
        for i in range(n_trials):
            auc_trials[i] = 0.5
    else:
        srange = np.abs(smax - smin)
        smin -= (1e-3 * srange);
        smax += (1e-3 * srange);
        delta = (srange + 2e-3*srange) / n_steps
        for t in range(n_trials):

            balance = float(spike_cnt[t]) / n_samples

            if spike_cnt[t] > 0:

                for b in range(n_steps):
                    threshold = smin + b * delta;

                    for i in range(n_samples):
                        pred = score[i] + threshold;
                        if ((Y[i,t] > 0) and (pred > 0)):
                            tpr[b] += 1.;
                        if ((Y[i,t] <= 0) and (pred > 0)):
                            fpr[b] += 1.;

                    tpr[b] /= (float(n_samples) * balance)
                    fpr[b] /= (float(n_samples) * (1. - balance))

                auc = 0.
                for b in range(n_steps-1):
                    auc += (tpr[b] + tpr[b+1]) / 2. * (fpr[b+1] - fpr[b])

            else:
                auc = 0.5

            auc_trials[t] = auc;

    return auc_trials

