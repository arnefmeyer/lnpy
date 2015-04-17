
#ifndef _TRON_WRAPPER_H_
#define _TRON_WRAPPER_H_

#include <math.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include "tron.h"
#include "helper.h"


static void tron_print_nothing(const char *s) {}

static void tron_print_something(const char *s)
{
	std::cout << s << std::endl;
}



/**************************************************************
* Learning methods
**************************************************************/

class TronFunction : public function
{
	public:

		TronFunction();
		~TronFunction();

		int get_nr_variable(void);
		virtual void fit(Problem *problem, double *w, double tolerance=1e-2, int max_iter=100, int verbose=0) = 0;

	protected:

		double *z;
		double *D;

		Problem *prob;
};


class BernoulliGLM : public TronFunction
{

	public:
		BernoulliGLM(Prior *p = NULL);

		void set_prior(Prior *p);
		Prior* get_prior();

		void fit(Problem *problem, double *w, double tolerance=1e-2, int max_iter=100, int verbose=0);

		double fun(double *w);
		void grad(double *w, double *g);
		void Hv(double *w, double *s, double *Hs);

	protected:
		Prior *prior;
};


class SVM : public TronFunction
{

	public:
		SVM(Prior *p = NULL);

		void set_prior(Prior *p);
		Prior* get_prior();

		void fit(Problem *problem, double *w, double tolerance=1e-2, int max_iter=100, int verbose=0);

		double fun(double *w);
		void grad(double *w, double *g);
		void Hv(double *w, double *s, double *Hs);

	protected:
		int *I;
		int sizeI;
		Prior *prior;
};


class PoissonGLM : public TronFunction
{

	public:
		PoissonGLM(Prior *p = NULL, double dt = 1., bool canonical = true);

		void set_prior(Prior *p);
		Prior* get_prior();

		void fit(Problem *problem, double *w, double tolerance=1e-2, int max_iter=100, int verbose=0);

		double fun(double *w);
		void grad(double *w, double *g);
		void Hv(double *w, double *s, double *Hs);

	protected:
		Prior *prior;
		double *rate;
		double dt;
		bool canonical;

		void calc_rate(double *z, int N);
};


class GaussianGLM : public TronFunction
{
	public:
		GaussianGLM(Prior *p = NULL);

		void set_prior(Prior *p);
		Prior* get_prior();

		void fit(Problem *problem, double *w, double tolerance=1e-2, int max_iter=100, int verbose=0);

		double fun(double *w);
		void grad(double *w, double *g);
		void Hv(double *w, double *s, double *Hs);

	protected:
		Prior *prior;
};

#endif

