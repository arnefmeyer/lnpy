lnpy
====

lnpy is a Python module for fitting parameters of neuronal stimulus-response 
functions (SRFs; for a review see [1]). While initially developed to estimate 
sensory neuron's receptive field parameters using a linear-nonlinear (LN)
model there are a number of methods to fit other models, e.g., linear and 
multilinear models.

This module is build on top of several excellent Python modules, e.g., 
SciPy, NumPy, matplotlib, scikits-learn, statsmodels, and neo. The critical 
parts are written in C(++) and wrapped using Cython.

The main focus is on auditory data and there are several submodules that 
facilitate reading and converting data. However, the estimators may just 
as well be applied to data from different modalities, e.g., the visual or 
somatosensory system.


Included estimators
===================

* Ridge regression (Machens et al. (2004))
* Automatic smoothness determination (ASD, Sahani & Linden (2003))
* Automatic locality determination (ALD, Park & Pillow (2011))
* Spike-triggered average (STA, deBoer & Kuyper (1968))
* Normalized reverse correlation (NRC, Theunissen et al. (2000))
* Generalized linear model (GLM, Paninski (2004), Truccolo et al. (2005))
* Maximum informative dimensions (MID, Sharpee et al. (2004)); requires external software
* Classification-based receptive field estimation (CbRF, Meyer et al. (2014a))
* Time-varying RF estimation using non-zero mean priors (Meyer et al. (2014b))
* Stochastic gradient descent-based versions of some of the above 
estimators (Meyer et al. (2015))

Moreover, the toolbox has been designed such that most models can be used with
a number of different priors, e.g., Gaussian and Laplace priors, and also 
mixtures of these priors.


Dependencies
============

lnpy works under Python 2.7 (Python 2.6 and Python 3.x have not been tested) and requires a working C/C++ compiler.

The required dependencies to build the package are

	* NumPy >= 1.6.2
	* SciPy >= 0.12
	* cython
	* matplotlib >= 1.2.1
	* statsmodels >= 0.4.2
	* neo >= 0.3
	* scikit-learn >= 0.14
	* sphinx
	* sphinxcontrib-napoleon


Installation
============

Linux
-----

1. Install the latest version of Python 2.7

2. Install dependencies, e.g.,

	pip install numpy scipy matplotlib neo

3. Install package by changing into the package directory and typing

	python setup.py install

   If you just want to compile the code in the package directory use

	python setup.py install build_ext -i


Windows 7
---------

1. Install the latest version of WinPython 2.7.x.x (64 bit version)

	http://sourceforge.net/projects/winpython/

	I haven't tested Python(x,y) but it should also work. 


2. Install the Microsoft Visual C++ 2008 Express edition

	http://download.microsoft.com/download/A/5/4/A54BADB6-9C3F-478D-8657-93B3FC9FE62D/vcsetup.exe


3. Install Microsoft Windows SDK for Windows 7 and .NET Framework 3.5 SP1 

	http://www.microsoft.com/en-us/download/details.aspx?id=3138

	Note: this gives you the 64 bit compiler. However, you don't have to install all components.
	Just uncheck everything except "Developer Tools >> Visual C++ Compilers" (and the header files).


4. To install the package, open the WinPython command prompt ("WinPython Command Prompt.exe" in WinPython directory) and type

	"XXX:\\Program Files (x86)\\Microsoft Visual Studio 9.0\\Common7\\Tools\\vsvars64.bat"

	where "XXX" denotes your system's root directory (probably "C")


5. Install dependencies; open the WinPython command prompt ("WinPython Command Prompt.exe" in WinPython directory) and type

	pip install neo


6. Install package by changing into the package directory and typing

	python setup.py install


That's all!


Some useful links
=================

* Anaconda Python (includes many scientific packages; free version available)
	http://docs.continuum.io/anaconda/index.html

* Matlab-like IDE: Spyder (included in Anaconda and WinPython)
	https://pythonhosted.org/spyder/

* Scientific programming in Python:
	http://www.scipy.org/

* Switching from Matlab to Python (using numpy package):
	http://wiki.scipy.org/NumPy_for_Matlab_Users
	http://mathesaurus.sourceforge.net/matlab-numpy.html

* Plotting in Python:
	http://matplotlib.org/


External software
=================

* The trust region conjugate gradient descent algorithm as described in [8] has been adapted from the liblinear library available at:

	http://www.csie.ntu.edu.tw/~cjlin/liblinear/

	Copyright (c) 2007-2013 The LIBLINEAR Project.
	All rights reserved.

* The MID algorithm requires additional code available at

	https://github.com/sharpee/mid

* The gammatone filterbank is based on Volker Hohmann's implementation (cf. [9]) available at

	http://www.uni-oldenburg.de/medizin/departments/mediphysik-akustik/mediphysik/downloads/

  You have to download the code and unpack Gfb_analyze.h and Gfb_analyze.h to
  lnpy/transform/code/gammatone.


References
==========
A.F. Meyer, R.S. Williamson, J.F. Linden, M. Sahani. Models of neuronal stimulus-response functions: elaboration, estimation and evaluation. Front. Syst. Neurosci. 2017.

M. Sahani & J.F. Linden. Evidence Optimization Techniques for Estimating Stimulus-Response Functions. NIPS, 2003, 317-324.

M. Park & J.W. Pillow. Receptive field inference with localized priors. PLoS Comput Biol, 2011, 7, e1002219.

E. deBoer & P. Kuyper. Triggered Correlation. IEEE Transactions on Biomedical Engineering, BM15, 169-179, 1968.

F.E. Theunissen, K. Sen, A.J. Doupe. Spectral-temporal receptive fields of nonlinear auditory neurons obtained using natural sounds. J Neurosci, 20, 2315-2331, 2000.

C.K. Machens, M.S. Wehr, A.M. Zador. Linearity of cortical receptive fields measured with natural sounds. J Neurosci, 24, 1089-1100, 2004.

L. Paninski. Maximum likelihood estimation of cascade point-process neural encoding models. Network, 15, 243-262, 2004.

W. Truccolo, U.T. Eden, M.R. Fellows, J.P. Donoghue, E.N. Brown. A point process framework for relating neural spiking activity to spiking history, neural ensemble, and extrinsic covariate effects. J Neurophysiol, 93, 1074-1089, 2005.

T. Sharpee, N.C. Rust, W. Bialek. Analyzing neural responses to natural signals: maximally informative dimensions. Neural Comput, 16, 223-250, 2004.

A.F. Meyer AF, J.P Diepenbrock, M.F. Happel, F.W. Ohl, J. Anemüller. Discriminative Learning of Receptive Fields from Responses to Non-Gaussian Stimulus Ensembles. PLOS ONE, 9, e93062, 2014.

A.F. Meyer AF, J.P Diepenbrock, M.F. Happel, F.W. Ohl, J. Anemüller. Temporal variability of spectro-temporal receptive fields in the anesthetized auditory cortex. Front Comput Neurosci, 8, 165, 2014.

A.F. Meyer AF, J.P Diepenbrock, M.F. Happel, F.W. Ohl, J. Anemüller. Fast and robust estimation of spectro-temporal receptive fields using stochastic approximations. J Neurosci Methods, 246, 119-133, 2015.

C.J. Lin, R.C. Weng, S.S. Keerthi. Trust Region Newton Method for Logistic Regression J. Mach. Learn. Res., 9, 627-650, 2008.

V. Hohmann. Frequency analysis and synthesis using a Gammatone filterbank ACTA ACUSTICA UNITED WITH ACUSTICA, 88, 433-442, 2002.
