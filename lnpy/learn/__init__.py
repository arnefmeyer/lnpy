"""
The :mod:`lnpy.learn` module implements several generalized linear models
(GLMs) and support vector machine classifiers (SVMs). Both batch gradient
descent and stochastic gradient descent solver are available. Most of the
implementations are wrappers around highly efficient C++ code.
"""

#from .pyhelper import PyProblem as Problem
from .base import SparseProblem, DenseProblem
from .pyhelper import PyGaussianPrior as GaussianPrior
from .pyhelper import PyLaplacePrior as LaplacePrior
from .pyhelper import PyENetPrior as ENetPrior
from .pyhelper import PyMixedPrior as MixedPrior
from .pyhelper import PySmoothnessPrior as SmoothnessPrior

from .pyhelper import PyGaussianLoss as GaussianLoss
from .pyhelper import PyLogLoss as LogLoss
from .pyhelper import PyHingeLoss as HingeLoss
from .pyhelper import PySquaredHingeLoss as SquaredHingeLoss

#from .glm import BernoulliGLM, PoissonGLM

__all__ = ['SparseProblem',
           'DenseProblem',
           'GaussianPrior',
           'LaplacePrior',
           'ENetPrior',
           'MixedPrior',
           'SmoothnessPrior',
           'GaussianLoss',
           'LogLoss',
           'HingeLoss',
           'SquaredHingeLoss']
#           'BernoulliGLM',
#           'PoissonGLM']
