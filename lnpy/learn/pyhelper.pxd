
from libcpp.string cimport string
from libc.stdint cimport uint32_t, int64_t, uint64_t
from libcpp cimport bool


cdef extern from "src/code/helper.h":

    cdef cppclass Problem:
        int get_N()
        int get_ndim()
        int get_ntrials()
        void set_C(double *c)
#        void set_permutation(int k, unsigned long *ind)
        void set_permutation(int k, uint64_t *ind)
        void reset_index()
        double *get_Y()

    cdef cppclass SparseProblem(Problem):
        SparseProblem(double *X, double *Y, int N, int p, double bias, int n_trials,
                      double *C)

    cdef cppclass DenseProblem(Problem):
        DenseProblem(double *X, double *Y, int N, int p, double bias, int n_trials,
                     double *C)

    cdef cppclass LossFunction:
        pass

    cdef cppclass GaussianLoss(LossFunction):
        GaussianLoss()

    cdef cppclass LogLoss(LossFunction):
        LogLoss()

    cdef cppclass HingeLoss(LossFunction):
        HingeLoss()
		
    cdef cppclass SquaredHingeLoss(LossFunction):
        SquaredHingeLoss()

    cdef cppclass PoissonLoss(LossFunction):
        PoissonLoss(double dt, bool canonical)

    cdef cppclass Prior:
        string get_name()
        double get_alpha()
        void set_alpha(double alpha)
        int get_ndim()
        void set_ndim(int d)

    cdef cppclass GaussianPrior(Prior):
        GaussianPrior(double alpha, int ndim, int penalize_bias)
        GaussianPrior(double alpha, double *mu, int ndim, int penalize_bias)

    cdef cppclass LaplacePrior(Prior):
        LaplacePrior(double alpha)
        LaplacePrior(double alpha, int ndim)
        LaplacePrior(double alpha, int ndim, double eps)
        
    cdef cppclass MixedPrior(Prior):
        MixedPrior(Prior *p, Prior *q)
        MixedPrior(Prior *p, Prior *q, double alpha, double gamma)
        double get_gamma()
        void set_gamma(double gamma)
        Prior* get_prior_1()
        Prior* get_prior_2()

    cdef cppclass ElasticNetPrior(Prior):
        ElasticNetPrior(double alpha, double gamma, int ndim, double eps)
        double get_gamma()
        void set_gamma(double gamma)

    cdef cppclass SmoothnessPrior(Prior):
        SmoothnessPrior(double alpha, double gamma, double *D, int ndim)
        double get_gamma()
        void set_gamma(double gamma)



cdef extern from "src/code/wrap_tron.h":

    cdef cppclass TronFunction:
        int get_nr_variable()
        void fit(Problem *prob, double *w, double tolerance, int max_iter,
                 int verbose)
        
    cdef cppclass BernoulliGLM(TronFunction):
        BernoulliGLM(Prior *prior)
        Prior* get_prior()
        void set_prior(Prior *prior)

    cdef cppclass SVM(TronFunction):
        SVM(Prior *prior)
        Prior* get_prior()
        void set_prior(Prior *prior)
        
    cdef cppclass PoissonGLM(TronFunction):
        PoissonGLM(Prior *prior, double dt, bool canonical)
        Prior* get_prior()
        void set_prior(Prior *prior)

    cdef cppclass GaussianGLM(TronFunction):
        GaussianGLM(Prior *prior)
        Prior* get_prior()
        void set_prior(Prior *prior)

cdef extern from "src/code/sgd.h":

    cdef cppclass SGD:
        SGD(double *v, int n_dim, double fit_bias, LossFunction *loss,
            double n_epochs, double alpha, bool warm_start)
        int fit(Problem *prob, int start_iter, int verbose)
        double* get_weights()
        double get_bias()
        int get_ndims()
        void set_alpha(int alpha)
        double get_alpha()

    cdef cppclass ASGD:
        ASGD(double *v, double *va, int n_dim, double fit_bias,
             LossFunction *loss, double n_epochs, double alpha,
             bool warm_start, double poly_decay)
        int fit(Problem *prob, int start_iter, int verbose)
        double* get_weights()
        double get_bias()
        int get_ndims()
        void set_alpha(int alpha)
        double get_alpha()

