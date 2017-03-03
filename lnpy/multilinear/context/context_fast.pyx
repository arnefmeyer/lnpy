# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

#cython: cdivision=True
#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np

ctypedef np.float64_t DOUBLE
ctypedef np.int64_t INT

cdef extern double ddot_(int *N, double *X, int *INCX, double *Y, int *INCY)
cdef extern int dscal_(int *n, double *sa, double *sx, int *incx)
cdef extern int daxpy_(int *n, double *sa, double *sx, int *incx, double *sy,
                       int *incy)

np.import_array()


# https://jakevdp.github.io/blog/2012/08/08/memoryview-benchmarks/
# http://docs.cython.org/src/userguide/memoryviews.html


""" ***********************************************************************
Alternating least squares implementation
*********************************************************************** """

def compute_A_matrix(double[:, ::1] S not None, double[:] w not None,
                     double[:, ::1] A not None, int T,
                     int J, int K, int M, int N, double c2=1.):

    cdef int i, j, k, m, n, N_tot, pad_len
    cdef double a

    N_tot = 2*N+1
    pad_len = J+M-1

    cdef int incr = 1

    for i in range(T):

        for j in range(J):
            for k in range(K):

                a = 0.
                for m in range(M+1):
                    for n in range(-N, N+1):
                        a += w[m*N_tot + N+n] * S[pad_len+i-j-m, N+k+n]

                # remove CGFs own context
                a -= w[N] * S[pad_len+i-j, N+k]
#                    a += ddot_(&N_tot, &w[m*N_tot], &incr,
#                               &S[pad_len+i-j-m, k], &incr)

                A[i, j*K + k] = S[pad_len+i-j, N+k] * (c2 + a)



def compute_B_matrix(double[:, ::1] S, double[:] w,
                     double[:, ::1] B, int T,
                     int J, int K, int M, int N):

    cdef int i, j, k, m, n, N_tot, pad_len
    cdef double b

    N_tot = 2*N+1
    pad_len = J+M-1

    for i in range(T):

        for m in range(M+1):
            for n in range(-N, N+1):

                b = 0.
                for j in range(J):
                    for k in range(K):
                        b += w[j*K+k] * S[pad_len+i-j, N+k] *\
                            S[pad_len+i-j-m, N+k+n]

                B[i, m*N_tot + N+n] = b


def predict_y_prf(double[:, ::1] S_pad, double[:] w, double[:] y_prf,
                 int T, int J, int K, int M, int N):

    cdef int i, j, k, pad_len

    pad_len = J+M-1

    for i in range(T):
        for j in range(J):
            for k in range(K):
                y_prf[i] += w[j*K + k] * S_pad[pad_len+i-j, N+k]


def predict_y_context(double[:, ::1] S_pad, double[:] w_prf, double[:] w_cgf,
                      double[:] y_pred, double c1, double c2, double c3,
                      int T, int J, int K, int M, int N):

    cdef int i, j, k, m, n
    cdef double a, b
    cdef int pad_len

    pad_len = J-1+M

    for i in range(T):

        a = c1

        for j in range(J):
            for k in range(K):

                # CGF part
                b = c3
                for m in range(M+1):
                    for n in range(-N, N+1):
                        b += w_cgf[m*(2*N+1)+N+n] * S_pad[pad_len+i-j-m, N+k+n]

                b -= w_cgf[N] * S_pad[pad_len+i-j, N+k]

                # PRF part
                a += w_prf[j*K + k] * S_pad[J-1+M+i-j, N+k] * (c2 + b)

        y_pred[i] += a


""" ***********************************************************************
Variational Bayesian EM implementation (not working yet)
*********************************************************************** """

def e_step_prf_compute_C(double[:, ::1] S not None, double[:] mu not None,
                         double[:, ::1] Sigma not None,
                         double[:, ::1] C not None, int T,
                         int J, int K, int M, int N, double c2=1.):

    """eq 30"""
    pass


def e_step_cgf_compute_C(double[:, ::1] S not None, double[:] mu not None,
                         double[:, ::1] Sigma not None,
                         double[:, ::1] C not None, int T,
                         int J, int K, int M, int N, double c2=1.):

    """eq 30"""
    pass


def e_step_prf_compute_v(double[:, ::1] S not None, double[:] mu not None,
                         double[:] v not None, int T,
                         int J, int K, int M, int N):

    """eq 34"""
    pass


def e_step_cgf_compute_v(double[:, ::1] S not None, double[:] mu not None,
                         double[:] v not None, int T,
                         int J, int K, int M, int N):

    """eq 34"""
    pass


def e_step_cgf_compute_noisevar(double[:, ::1] S_pad not None,
                                double [:] y not None,
                                double[:] mu_prf not None,
                                double[:] mu_cgf not None,
                                double [:, ::1] Sigma_prf not None,
                                double [:, ::1] Sigma_cgf not None,
                                double[:] v not None, int T,
                                int J, int K, int M, int N):

    """eq 41 + eq 42"""

    cdef double noisevar
    cdef double t1 = 0
    cdef double t2 = 0
    cdef int i, j, k, m, n, ii, jj, kk, mm, nn
    cdef double yTy = 0

    cdef int N_tot = 2*N+1
    cdef int pad_len = J+M-1

    cdef double c1, c2, c3
    cdef double a, b
    cdef double u_a, u_aa, u_b, u_bb, s_a, s_aa, s_b, s_bb

    c1 = mu_prf[J*K]
    c2 = 1
    c3 = 0

    for i in range(T):

        # y'y
        yTy += y[i]*y[i]

        # t1 term
        a = c1
        for j1 in range(J):
            for k1 in range(K):

                b = c3
                for m in range(M+1):
                    for n in range(-N, N+1):
                        b += mu_cgf[m*(2*N+1)+N+n] * S_pad[pad_len+i-j-m, N+k+n]

                a += mu_prf[j*K+k] * S_pad[J-1+M+i-j, N+k] * (c2 + b)

        t1 += y[i] * a

        # t2 term
        a = 0
        for j in range(J):
            for k in range(K):

                u_a = mu_prf[j*K+k]
                s_a = S_pad[pad_len+i-j, N+k]

                for jj in range(J):
                    for kk in range(K):

                        u_aa = u_a * mu_prf[jj*K+kk]
                        u_aa += Sigma_prf[j*K+k, jj*K+kk]

                        s_aa = S_pad[pad_len+i-jj, N+kk]

                        b = c3
                        for m in range(M+1):
                            for n in range(-N, N+1):
                                u_b = mu_cgf[m*(2*N+1)+N+n]

                                s_b = S_pad[pad_len+i-j-m, N+k+n]

                                for mm in range(M+1):
                                    for nn in range(-N, N+1):
                                        u_bb = u_b * mu_cgf[mm*(2*N+1)+N+nn]
                                        u_bb += Sigma_cgf[m*(2*N+1)+N+n, mm*(2*N+1)+N+nn]

                                        s_bb = S_pad[pad_len+i-jj-mm, N+kk+nn]

                                        a += u_aa * s_a * s_aa * u_bb * s_bb
#                                        b += mu_cgf[m*(2*N+1)+N+n] * S_pad[pad_len+i-j-m, N+k+n]
        
#                        a += mu_prf[j*K+k] * * S_pad[J-1+M+i-j, N+k] * (c2 + b)

        t2 += a

    noisevar = (yTy - 2 * t1 + t2)

    return noisevar
