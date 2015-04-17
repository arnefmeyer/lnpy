#!python
#cython: boundscheck=False
#cython: wraparound=False

import  numpy as np
cimport numpy as np
cimport pyhelper

np.import_array()


cdef class PyProblem:
    cdef Problem *thisptr

    def __cinit__(self):
        pass
    
    def __dealloc__(self):
        pass

    cdef Problem* get_pointer(self):
        return self.thisptr

    def get_N(self):
        return self.thisptr.get_N()

    def get_ndim(self):
        return self.thisptr.get_ndim()

    def get_ntrials(self):
        return self.thisptr.get_ntrials()        

    def get_Y(self):
        cdef double *ptr = self.thisptr.get_Y()
        cdef np.npy_intp shape[2]
        shape[0] = self.get_N()
        shape[1] = self.get_ntrials()
        arr = np.PyArray_SimpleNewFromData(2, shape, np.NPY_DOUBLE, ptr) 

        return arr

    def set_weights(self, np.ndarray[np.float64_t, ndim=1, mode='c'] C):
        if C is None:
            self.thisptr.set_C(NULL)
        else:
            self.thisptr.set_C(<double*>C.data)

    def set_permutation(self, np.ndarray[np.uint64_t, ndim=1, mode='c'] perm):
        if perm is None:
            self.thisptr.set_permutation(0, NULL)
        else:
            self.thisptr.set_permutation(perm.shape[0], <uint64_t*>perm.data)

    def reset_index(self):
        self.thisptr.reset_index()


cdef class PySparseProblem(PyProblem):
    
    def __cinit__(self, np.ndarray[np.float64_t, ndim=2, mode='c'] X,
                 np.ndarray[np.float64_t, ndim=1, mode='c'] Y,
                 double bias=1,
                 np.ndarray[np.float64_t, ndim=1, mode='c'] C=None, **kwargs):

        cdef double *cp
        if C is None:
            cp = <double*>NULL
        else:
            cp = <double*>C.data

        cdef int n_trials = Y.shape[0] / X.shape[0] 

        self.thisptr = new SparseProblem(<double*>X.data, <double*>Y.data,
                                         X.shape[0], X.shape[1],
                                         bias, n_trials, cp)

    def __dealloc__(self):
        del self.thisptr


cdef class PyDenseProblem(PyProblem):
    
    def __cinit__(self, np.ndarray[np.float64_t, ndim=2, mode='c'] X,
                 np.ndarray[np.float64_t, ndim=1, mode='c'] Y,
                 double bias=1,
                 np.ndarray[np.float64_t, ndim=1, mode='c'] C=None, **kwargs):

        cdef double *cp
        if C is None:
            cp = <double*>NULL
        else:
            cp = <double*>C.data

        cdef int n_trials = Y.shape[0] / X.shape[0]

        self.thisptr = new DenseProblem(<double*>X.data, <double*>Y.data,
		                              X.shape[0], X.shape[1],
                                        bias, n_trials, cp)

    def __dealloc__(self):
        del self.thisptr
    

cdef class PyLossFunction:
    cdef readonly int binary_targets

    def __cinit__(self):
        self.binary_targets = 0
    
    def __dealloc__(self):
        pass

    cdef LossFunction* get_pointer(self):
        return <LossFunction*>NULL


cdef class PyGaussianLoss(PyLossFunction):
    cdef GaussianLoss *thisptr

    def __cinit__(self):
        self.thisptr = new GaussianLoss()
        self.binary_targets = 0

    def __dealloc__(self):
        del self.thisptr

    cdef LossFunction* get_pointer(self):
        return self.thisptr


cdef class PyLogLoss(PyLossFunction):
    cdef LogLoss *thisptr

    def __cinit__(self):
        self.thisptr = new LogLoss()
        self.binary_targets = 1

    def __dealloc__(self):
        del self.thisptr

    cdef LossFunction* get_pointer(self):
        return self.thisptr


cdef class PyHingeLoss(PyLossFunction):
    cdef HingeLoss *thisptr

    def __cinit__(self):
        self.thisptr = new HingeLoss()
        self.binary_targets = 1

    def __dealloc__(self):
        del self.thisptr

    cdef LossFunction* get_pointer(self):
        return self.thisptr


cdef class PySquaredHingeLoss(PyLossFunction):
    cdef SquaredHingeLoss *thisptr

    def __cinit__(self):
        self.thisptr = new SquaredHingeLoss()
        self.binary_targets = 1

    def __dealloc__(self):
        del self.thisptr

    cdef LossFunction* get_pointer(self):
        return self.thisptr


cdef class PyPoissonLoss(PyLossFunction):
    cdef PoissonLoss *thisptr

    def __cinit__(self, double dt=1., int canonical=1):
        self.thisptr = new PoissonLoss(dt, canonical > 0)

    def __dealloc__(self):
        del self.thisptr

    cdef LossFunction* get_pointer(self):
        return self.thisptr



cdef class PyPrior:
    cdef Prior *thisptr

    def __cinit__(self):
        pass
    
    def __dealloc__(self):
        pass

    cdef Prior* get_pointer(self):
        return self.thisptr

    property name:
        def __get__(self):
            return self.__class__.__name__

    property alpha:
        def __get__(self):
            return self.thisptr.get_alpha()

        def __set__(self, x):
            self.thisptr.set_alpha(x)

    def get_ndim(self):
        return self.thisptr.get_ndim()

    def set_ndim(self, int d):
        self.thisptr.set_ndim(d)

    def get_default_grid(self, n_alpha=10, alpha_range=(2.**-20, 2.**20),
                         n_gamma=10, gamma_range=(0., 1.)):

        param_grid = {}
        param_info = {}

        if hasattr(self, 'alpha'):
            values = 2. ** np.linspace(np.log2(alpha_range[0]),
                                       np.log2(alpha_range[1]), n_alpha)
            param_grid.update({'alpha': values})
            param_info.update({'alpha': {'scaling': 'log2'}})

        if hasattr(self, 'gamma'):
            values = np.linspace(gamma_range[0], gamma_range[1], n_gamma)
            param_grid.update({'gamma': values})
            param_info.update({'gamma': {'scaling': 'linear'}})

        return param_grid, param_info


cdef class PyGaussianPrior(PyPrior):

    def __cinit__(self, double alpha=0, int ndim=-1, int penalize_bias=-1,
                  np.ndarray[np.float64_t, ndim=1, mode='c'] mu=None):
        if mu is None:
            self.thisptr = new GaussianPrior(alpha, ndim, penalize_bias)
        else:
            self.thisptr = new GaussianPrior(alpha, <double*>mu.data,
                                             mu.shape[0], penalize_bias)
    
    def __dealloc__(self):
        del self.thisptr


cdef class PyLaplacePrior(PyPrior):

    def __cinit__(self, double alpha=0, int ndim=-1, double eps=1e-8):
        self.thisptr = new LaplacePrior(alpha, ndim, eps)
    
    def __dealloc__(self):
        del self.thisptr


cdef class PyMixedPrior(PyPrior):

    def __cinit__(self, PyPrior p, PyPrior q, double alpha=1, double gamma=.5):
        self.thisptr = new MixedPrior(p.get_pointer(), q.get_pointer(),
                                      alpha, gamma)

    def __dealloc__(self):
        del self.thisptr

    def __reduce__(self):
        cdef MixedPrior *cp
        cp = <MixedPrior*>self.thisptr

        cpdef float alpha = cp.get_alpha()
        cpdef float gamma = cp.get_gamma()
        cdef Prior *tp1 = <Prior*>cp.get_prior_1()
        cpdef PyPrior p1 = PyPrior()
        p1.thisptr = tp1
        cdef Prior *tp2 = <Prior*>cp.get_prior_2()
        cpdef PyPrior p2 = PyPrior()
        p2.thisptr = tp2

        dd = (rebuild_mixed_prior, (p1, p2, alpha, gamma))
        return dd    

    property gamma:
        def __get__(self):
            cdef MixedPrior *cp
            cp = <MixedPrior*>self.thisptr
            return cp.get_gamma()

        def __set__(self, x):
            cdef MixedPrior *cp = <MixedPrior*>self.thisptr
            cp.set_gamma(x)

def rebuild_mixed_prior(p1, p2, alpha, gamma):
    return PyMixedPrior(p1, p2, alpha, gamma)


cdef class PyENetPrior(PyPrior):

    def __cinit__(self, double alpha=1., double gamma=.5, int ndim=-1,
                  double eps=1e-12):
        self.thisptr = new ElasticNetPrior(alpha, gamma, ndim, eps)
    
    def __dealloc__(self):
        del self.thisptr

    property gamma:
        def __get__(self):
            cdef ElasticNetPrior *cp
            cp = <ElasticNetPrior*>self.thisptr
            return cp.get_gamma()

        def __set__(self, x):
            cdef ElasticNetPrior *cp = <ElasticNetPrior*>self.thisptr
            cp.set_gamma(x)


cdef class PySmoothnessPrior(PyPrior):
    cdef np.ndarray precision_mat

    def __cinit__(self, double alpha=0, double gamma=0,
                  np.ndarray[np.float64_t, ndim=2, mode='c'] D=None):
        self.thisptr = new SmoothnessPrior(alpha, gamma, <double*>D.data,
                                           D.shape[0])
        self.precision_mat = np.copy(D)
    
    def __dealloc__(self):
        del self.thisptr

    def __reduce__(self):
        cdef SmoothnessPrior *cp
        cp = <SmoothnessPrior*>self.thisptr

        cpdef float alpha = cp.get_alpha()
        cpdef float gamma = cp.get_gamma()
        cpdef int ndim = cp.get_ndim()
        
        cpdef np.ndarray[np.float64_t, ndim=2] D = self.precision_mat

        dd = (rebuild_smoothness_prior, (alpha, gamma, D))
        return dd        

    property alpha:
        def __get__(self):
            cdef SmoothnessPrior *cp
            cp = <SmoothnessPrior*>self.thisptr
            return cp.get_alpha()

        def __set__(self, x):
            cdef SmoothnessPrior *cp = <SmoothnessPrior*>self.thisptr
            cp.set_alpha(x)

    property gamma:
        def __get__(self):
            cdef SmoothnessPrior *cp
            cp = <SmoothnessPrior*>self.thisptr
            return cp.get_gamma()

        def __set__(self, x):
            cdef SmoothnessPrior *cp = <SmoothnessPrior*>self.thisptr
            cp.set_gamma(x)

def rebuild_smoothness_prior(alpha, gamma, D):
    return PySmoothnessPrior(alpha, gamma, D)


cdef class PyBernoulliGLM:
    cdef BernoulliGLM *thisptr
    cpdef PyPrior prior

    def __cinit__(self, PyPrior prior=None):
        if prior is None:
            prior = PyGaussianPrior()
        self.thisptr = new BernoulliGLM(<Prior*>prior.get_pointer())
        self.prior = prior

    def __dealloc__(self):
        del self.thisptr

    def get_prior(self):
        return self.prior

    def fit(self, PyProblem prob, np.ndarray[np.float64_t, ndim=1, mode='c'] w,
            double tolerance=1e-2, int max_iter=100, int verbose=0):
        self.thisptr.fit(<Problem*>prob.get_pointer(), <double*>w.data,
                         tolerance, max_iter, verbose)


cdef class PySVM:
    cdef SVM *thisptr
    cpdef PyPrior prior

    def __cinit__(self, PyPrior prior=None):
        if prior is None:
            prior = PyGaussianPrior()
        self.thisptr = new SVM(<Prior*>prior.get_pointer())
        self.prior = prior

    def __dealloc__(self):
        del self.thisptr

    def get_prior(self):
        return self.prior

    def fit(self, PyProblem prob, np.ndarray[np.float64_t, ndim=1, mode='c'] w,
            double tolerance=1e-2, int max_iter=100, int verbose=0):
        self.thisptr.fit(<Problem*>prob.get_pointer(), <double*>w.data,
                         tolerance, max_iter, verbose)


cdef class PyPoissonGLM:
    cdef PoissonGLM *thisptr
    cpdef PyPrior prior

    def __cinit__(self, PyPrior prior=None, double dt=1., int canonical=1):
        if prior is None:
            prior = PyGaussianPrior()
        self.thisptr = new PoissonGLM(<Prior*>prior.get_pointer(), dt,
                                      canonical > 0)
        self.prior = prior

    def __dealloc__(self):
        del self.thisptr

    def get_prior(self):
        return self.prior

    def fit(self, PyProblem prob, np.ndarray[np.float64_t, ndim=1, mode='c'] w,
            double tolerance=1e-2, int max_iter=100, int verbose=0):
        self.thisptr.fit(<Problem*>prob.get_pointer(), <double*>w.data,
                         tolerance, max_iter, verbose)


cdef class PyGaussianGLM:
    cdef GaussianGLM *thisptr
    cpdef PyPrior prior

    def __cinit__(self, PyPrior prior=None):
        if prior is None:
            prior = PyGaussianPrior()
        self.thisptr = new GaussianGLM(<Prior*>prior.get_pointer())
        self.prior = prior

    def __dealloc__(self):
        del self.thisptr

    def get_prior(self):
        return self.prior

    def fit(self, PyProblem prob, np.ndarray[np.float64_t, ndim=1, mode='c'] w,
            double tolerance=1e-2, int max_iter=100, int verbose=0):
        self.thisptr.fit(<Problem*>prob.get_pointer(), <double*>w.data,
                         tolerance, max_iter, verbose)


cdef class PySGD:
    cdef SGD *thisptr

    def __cinit__(self, np.ndarray[np.float64_t, ndim=1, mode='c'] w,
                  double bias, double n_epochs=1.,
                  PyLossFunction loss=None, double alpha=1.,
                  bool warm_start=False):
        cdef int n_dim
        n_dim = w.shape[0] - (int)(bias > 0)
        if loss is None:
            loss = PyGaussianLoss()
        self.thisptr = new SGD(<double*>w.data, n_dim, bias,
                               <LossFunction*>loss.get_pointer(),
                               n_epochs, alpha, warm_start)

    def __dealloc__(self):
        del self.thisptr

    def fit(self, PyProblem prob, int start_iter=0, int verbose=0):
        return self.thisptr.fit(<Problem*>prob.get_pointer(), start_iter,
                                verbose)

    def get_bias(self):
        return self.thisptr.get_bias()

    def set_alpha(self, int alpha):
        self.thisptr.set_alpha(alpha)

    def get_alpha(self):
        return self.thisptr.get_alpha()


cdef class PyASGD:
    cdef ASGD *thisptr

    def __cinit__(self, np.ndarray[np.float64_t, ndim=1, mode='c'] w,
                  np.ndarray[np.float64_t, ndim=1, mode='c'] wa, double bias,
                  double n_epochs=1., PyLossFunction loss=None,
                  double alpha=1., bool warm_start=False,
                  double poly_decay=2.):
        
        cdef int ndim
        ndim = w.shape[0] - (int)(bias > 0)
        if loss is None:
            loss = PyGaussianLoss()
        self.thisptr = new ASGD(<double*>w.data, <double*>wa.data, ndim,
                                bias, <LossFunction*>loss.get_pointer(),
                                n_epochs, alpha, warm_start, poly_decay)

    def __dealloc__(self):
        del self.thisptr

    def fit(self, PyProblem prob, int start_iter=0, int verbose=0):
        return self.thisptr.fit(<Problem*>prob.get_pointer(), start_iter,
                                verbose)

    def get_bias(self):
        return self.thisptr.get_bias()

    def set_alpha(self, int alpha):
        self.thisptr.set_alpha(alpha)

    def get_alpha(self):
        return self.thisptr.get_alpha()
