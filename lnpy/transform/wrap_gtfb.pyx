# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: arne.f.meyer@uni-oldenburg.de
#
# License:

import cython
import numpy as np
cimport numpy as np
cimport cython
cimport wrap_gtfb

np.import_array()

ctypedef np.double_t DOUBLE

                    
cpdef process(Py_ssize_t num_bands,
              Py_ssize_t gamma_order,
              np.ndarray[DOUBLE, ndim=1, mode='c'] coef_real,
              np.ndarray[DOUBLE, ndim=1, mode='c'] coef_imag,
              np.ndarray[DOUBLE, ndim=1, mode='c'] norm_fact,
              np.ndarray[DOUBLE, ndim=1, mode='c'] states_real,
              np.ndarray[DOUBLE, ndim=1, mode='c'] states_imag,
              np.ndarray[DOUBLE, ndim=1, mode='c'] signal):
    """Simple wrapper for gammatone analyzer C code"""
    
    # Get data dimensions
    cdef Py_ssize_t N = signal.shape[0]

    # Create output arrays  
    cdef np.ndarray[DOUBLE, ndim=2, mode='c'] out_real = np.empty((N, num_bands), np.double, order='c')
    cdef np.ndarray[DOUBLE, ndim=2, mode='c'] out_imag = np.empty((N, num_bands), np.double, order='c')

	# Call C function 
    analyze(num_bands,
            gamma_order,
            N,
            <double*>coef_real.data,
            <double*>coef_imag.data,
            <double*>norm_fact.data,
            <double*>states_real.data,
            <double*>states_imag.data,
            <double*>signal.data,
            <double*>out_real.data,
            <double*>out_imag.data)

    return out_real + 1j * out_imag
            
