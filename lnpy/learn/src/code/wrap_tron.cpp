
#include "wrap_tron.h"


/***********************************************************************

	TRON function base class

***********************************************************************/

TronFunction::TronFunction()
{
}

TronFunction::~TronFunction()
{
}

int TronFunction::get_nr_variable(void)
{
	int d = this->prob->get_ndim() + (int)this->prob->get_fit_bias();
	return d;
}



/***********************************************************************

	Bernoulli GLM (with canonical link function)

***********************************************************************/

BernoulliGLM::BernoulliGLM(Prior *prior)
{
	this->prior = prior;
}

Prior* BernoulliGLM::get_prior()
{
	return this->prior;
}

void BernoulliGLM::set_prior(Prior *p)
{
	if (this->prior != NULL)
		delete this->prior;

	this->prior = p;
}

void BernoulliGLM::fit(Problem *problem, double *w, double tolerance, int max_iter, int verbose)
{
	TRON *solver;
	int N;

	this->prob = problem;
	N = prob->get_N() * prob->get_ntrials();
	z = new double[N];
	D = new double[N];

	solver = new TRON(this, tolerance, max_iter);
    if (verbose <= 0)
        solver->set_print_string(&tron_print_nothing);
	else
		solver->set_print_string(&tron_print_something);

	solver->tron(w);

	delete solver;
	delete[] z;
	delete[] D;
}

double BernoulliGLM::fun(double *w)
{

	int i;
	double f = 0;
	double *y = prob->get_Y();
	double *C = prob->get_C();
	int N = prob->get_N() * prob->get_ntrials();

	Prior *prior = this->prior;
	int n_dim = prob->get_ndim();

	prob->Xv(w, z);

	for(i=0;i<N;i++)
	{
		double yz = y[i]*z[i];
		if (yz >= 0)
			f += C[i] * log(1 + exp(-yz));
		else
			f += C[i] * (-yz+log(1 + exp(yz)));
	}

//	if (prob->get_fit_bias())
//		f += prob->get_bias();

	if (prior != NULL)
		f += prior->fun(w, n_dim);

	return(f);
}

void BernoulliGLM::grad(double *w, double *g)
{
	int i;
	int N = prob->get_N() * prob->get_ntrials();
	double *y = prob->get_Y();
	double *C = prob->get_C();

	Prior *prior = this->prior;
	int n_dim = prob->get_ndim();

	for(i=0;i<N;i++)
	{
		z[i] = 1/(1 + exp(-y[i]*z[i]));
		D[i] = z[i]*(1-z[i]);
		z[i] = C[i] * (z[i]-1)*y[i];
	}
	prob->XTv(z, g);

//	if (prob->get_fit_bias())
//		g[n_dim] += w[n_dim];

	if (prior != NULL)
		prior->grad(w, g, n_dim);
}

void BernoulliGLM::Hv(double *w, double *s, double *Hs)
{
	int i;
	int N = prob->get_N() * prob->get_ntrials();
	double *C = prob->get_C();
	double *wa = new double[N];

	Prior *prior = this->prior;
	int n_dim = prob->get_ndim();

	prob->Xv(s, wa);
	for(i=0;i<N;i++)
		wa[i] = C[i] * D[i] * wa[i];

	prob->XTv(wa, Hs);

//	if (prob->get_fit_bias())
//		Hs[n_dim] += s[n_dim];

	if (prior != NULL)
		prior->Hv(w, s, Hs, n_dim);

	delete[] wa;
}




///***********************************************************************

//	L2 loss support vector machine (SVM) classifier

//***********************************************************************/


SVM::SVM(Prior *prior)
{
	this->prior = prior;
}

Prior* SVM::get_prior()
{
	return this->prior;
}

void SVM::set_prior(Prior *p)
{
	if (this->prior != NULL)
		delete this->prior;

	this->prior = p;
}

void SVM::fit(Problem *problem, double *w, double tolerance, int max_iter, int verbose)
{
	TRON *solver;
	int N;

	this->prob = problem;
	N = prob->get_N() * prob->get_ntrials();
	z = new double[N];
	D = new double[N];
	I = new int[N];

	solver = new TRON(this, tolerance, max_iter);
    if (verbose <= 0)
        solver->set_print_string(&tron_print_nothing);
	else
		solver->set_print_string(&tron_print_something);

	solver->tron(w);

	delete solver;
	delete[] z;
	delete[] D;
	delete[] I;
}

double SVM::fun(double *w)
{
	int i;
	double f = 0;
	double *y = prob->get_Y();
	double *C = prob->get_C();
	int N = prob->get_N() * prob->get_ntrials();

	Prior *prior = this->prior;
	int n_dim = prob->get_ndim();

	prob->Xv(w, z);

	for(i=0;i<N;i++)
	{
		z[i] = y[i] * z[i];
		double d = 1 - z[i];
		if (d > 0)
			f += C[i] * d * d;
	}
	f /= 2.;

	if (prior != NULL) {
		f += prior->fun(w, n_dim);
	}

	return(f);
}

void SVM::grad(double *w, double *g)
{
	int n_trials = prob->get_ntrials();
	int N = prob->get_N() * n_trials;
	double *y = prob->get_Y();
	double *C = prob->get_C();

	Prior *prior = this->prior;
	int n_dim = prob->get_ndim();

	sizeI = 0;
	for (int i=0;i<N;i++)
		if (z[i] < 1)
		{
			z[sizeI] = C[i] * y[i]*(z[i]-1);
			I[sizeI] = i / n_trials;
			sizeI++;
		}

	prob->subXTv(z, g, I, sizeI);

	if (prior != NULL)
		prior->grad(w, g, n_dim);
}

void SVM::Hv(double *w, double *s, double *Hs)
{
	double *C = prob->get_C();

	Prior *prior = this->prior;
	int n_dim = prob->get_ndim();

	double *wa = new double[sizeI];
	prob->subXv(s, wa, I, sizeI);
	for(int i=0;i<sizeI;i++)
		wa[i] = C[I[i]] * wa[i];
	prob->subXTv(wa, Hs, I, sizeI);

	if (prior != NULL)
		prior->Hv(w, s, Hs, n_dim);

	delete[] wa;
}




///***********************************************************************

//Poisson GLM (without post-spike filter)

//***********************************************************************/

PoissonGLM::PoissonGLM(Prior *prior, double bin_width, bool canonical)
{
	this->prior = prior;
	this->dt = bin_width;
	this->canonical = canonical;
}

Prior* PoissonGLM::get_prior()
{
	return this->prior;
}

void PoissonGLM::set_prior(Prior *p)
{
	if (this->prior != NULL)
		delete this->prior;

	this->prior = p;
}

void PoissonGLM::fit(Problem *problem, double *w, double tolerance, int max_iter, int verbose)
{
	TRON *solver;
	int N;

	this->prob = problem;
	N = prob->get_N() * prob->get_ntrials();
	z = new double[N];
	D = new double[N];
	rate = new double[N];

	solver = new TRON(this, tolerance, max_iter);
    if (verbose <= 0)
        solver->set_print_string(&tron_print_nothing);
	else
		solver->set_print_string(&tron_print_something);

	solver->tron(w);

	delete solver;
	delete[] z;
	delete[] D;
	delete[] rate;
}

double PoissonGLM::fun(double *w)
{
	double f = 0;
	double *y = prob->get_Y();
	int N = prob->get_N() * prob->get_ntrials();

	Prior *prior = this->prior;
	int n_dim = prob->get_ndim();

	prob->Xv(w, z);
	calc_rate(z, N);

	if (canonical)
		for(int i=0;i<N;i++)
			f += rate[i] - y[i] * z[i];
	else
		for(int i=0;i<N;i++)
			f += rate[i] - y[i] * log(rate[i]);

	if (prior != NULL) {
		f += prior->fun(w, n_dim);
	}

//	std::cout << "f: " << f << std::endl;

	return(f);
}

void PoissonGLM::grad(double *w, double *g)
{
	int N = prob->get_N() * prob->get_ntrials();
	double *y = prob->get_Y();

	Prior *prior = this->prior;
	int n_dim = prob->get_ndim();

	if (canonical)
		for(int i=0;i<N;i++) {
			z[i] = rate[i] - y[i];
			D[i] = rate[i];
		}
	else {
		for(int i=0;i<N;i++) {
			if(z[i] <= 0) {	
				z[i] = (dt*rate[i] - y[i]);
				D[i] = dt*rate[i];
			}
			else {
				double zi = z[i];
				z[i] = (1 + zi) * (y[i] / (1 + zi + .5*zi*zi) - dt);

				/* Diagonal in Hessian term */
				double a = (1 + zi)*(1 + zi);
				double b = (1 + zi + .5*zi*zi);
				D[i] = -dt + y[i] * (1 - a/(b*b));
			}
		}
	}

	prob->XTv(z, g);

	if (prior != NULL)
		prior->grad(w, g, n_dim);
}

void PoissonGLM::Hv(double *w, double *s, double *Hs)
{
	int N = prob->get_N() * prob->get_ntrials();
	double *wa = new double[N];

	Prior *prior = this->prior;
	int n_dim = prob->get_ndim();

	prob->Xv(s, wa);
	for(int i=0;i<N;i++)
		wa[i] = D[i] * wa[i];

	prob->XTv(wa, Hs);

	if (prior != NULL)
		prior->Hv(w, s, Hs, n_dim);

	delete[] wa;
}

void PoissonGLM::calc_rate(double *z, int N)
{
	if (this->canonical)
		for(int i=0; i<N; i++)
			rate[i] = MAX(exp(z[i]), 1e-50);
	else {
		for(int i=0; i<N; i++) {
			if (z[i] <= 0)
				rate[i] = MAX(exp(z[i]), 1e-50);
			else
				rate[i] = 1 + z[i] + .5 * z[i] * z[i];
		}
	}
}


/***********************************************************************

	Linear (Gaussian) GLM (with canonical link function)

***********************************************************************/

GaussianGLM::GaussianGLM(Prior *prior)
{
	this->prior = prior;
}

Prior* GaussianGLM::get_prior()
{
	return this->prior;
}

void GaussianGLM::set_prior(Prior *p)
{
	if (this->prior != NULL)
		delete this->prior;

	this->prior = p;
}

void GaussianGLM::fit(Problem *problem, double *w, double tolerance, int max_iter, int verbose)
{
	TRON *solver;
	int N;

	this->prob = problem;
	N = prob->get_N() * prob->get_ntrials();
	z = new double[N];
	D = new double[N];

	solver = new TRON(this, tolerance, max_iter);
    if (verbose <= 0)
        solver->set_print_string(&tron_print_nothing);
	else
		solver->set_print_string(&tron_print_something);

	solver->tron(w);

	delete solver;
	delete[] z;
	delete[] D;
}

double GaussianGLM::fun(double *w)
{

	int i;
	double f = 0;
	double *y = prob->get_Y();
	int N = prob->get_N() * prob->get_ntrials();

	Prior *prior = this->prior;
	int n_dim = prob->get_ndim();

	prob->Xv(w, z);

	for(i=0;i<N;i++) {
		double dd = y[i] - z[i];
		f += dd * dd;
	}
	f /= 2;

	if (prior != NULL)
		f += prior->fun(w, n_dim);

	return(f);
}

void GaussianGLM::grad(double *w, double *g)
{
	int i;
	int N = prob->get_N() * prob->get_ntrials();
	double *y = prob->get_Y();

	Prior *prior = this->prior;
	int n_dim = prob->get_ndim();

	for(i=0;i<N;i++)
		z[i] = -(y[i] - z[i]);
	prob->XTv(z, g);

	if (prior != NULL)
		prior->grad(w, g, n_dim);
}

void GaussianGLM::Hv(double *w, double *s, double *Hs)
{
	int N = prob->get_N() * prob->get_ntrials();
	double *wa = new double[N];

	Prior *prior = this->prior;
	int n_dim = prob->get_ndim();

	prob->Xv(s, wa);
	prob->XTv(wa, Hs);

	if (prior != NULL)
		prior->Hv(w, s, Hs, n_dim);

	delete[] wa;
}


