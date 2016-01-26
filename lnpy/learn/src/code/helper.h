

#ifndef __TRON_HELPER_H__
#define __TRON_HELPER_H__

#include <vector>
#include <cmath>
#include <string>
#include <iostream>
#include <stdint.h>


#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))


/**************************************************
Use liblinear-like data format for sparse problem
**************************************************/

struct feature_node
{
	int index;
	double value;
};

//typedef unsigned long int index_t;
typedef uint64_t index_t;
typedef typename std::vector<feature_node*> FeatureMatrix;


class Problem
{
	protected:
		int N;
		int p;
		double *Y;
		double bias;
		int n_trials;
		double *C;
		bool delete_C;

		size_t perm_len;
		index_t *perm_ind;
		index_t current_pos;
		index_t current_index;

	public:
		virtual ~Problem() {}

		void init_problem(double *Y, int N, int p, double bias, int n_trials, double *C);
		void cleanup_problem();

		int get_N(){ return this->N; }
		int get_ndim() { return this->p; }
		int get_ntrials() { return this->n_trials; }
		
		double* get_Y(){ return this->Y; }
		double get_y(int index = -1);

		double get_bias(){ return this->bias; }
		void set_C(double *c);
		double* get_C(){ return this->C; }
		bool get_fit_bias() { return this->bias > 0; }

		void set_permutation(int k, index_t *ind);
		int next();
		index_t get_current_index();
		void reset_index();

		virtual bool is_sparse() = 0;
		virtual void* get_x(int index = -1) = 0;

		virtual double xv(double *v) = 0;

		virtual void Xv(double *v, double *Xv) = 0;
		virtual void XTv(double *v, double *XTv) = 0;

		virtual void subXv(double *v, double *Xv, int *I, int sizeI) = 0;
		virtual void subXTv(double *v, double *XTv, int *I, int sizeI) = 0;
};


class SparseProblem : public Problem
{
	protected:
		FeatureMatrix X;
		void repack_data_sparse(double *X, int N, int p, double bias, int n_trials);

	public:

		SparseProblem(double *X, double *Y, int N, int p, double bias, int n_trials, double *C);
		~SparseProblem();

		bool is_sparse() { return true; }
		void* get_X(){ return (void*)&this->X; }
		void* get_x(int index = -1);

		double xv(double *v);

		void Xv(double *v, double *Xv);
		void XTv(double *v, double *XTv);

		void subXv(double *v, double *Xv, int *I, int sizeI);
		void subXTv(double *v, double *XTv, int *I, int sizeI);		
};


class DenseProblem : public Problem
{
	protected:
		double *X;

	public:

		DenseProblem(double *X, double *Y, int N, int p, double bias, int n_trials, double *C);
		~DenseProblem();

		bool is_sparse() { return false; }

		void* get_X(){ return (void*)this->X; }
		void* get_x(int index = -1);

		double xv(double *v);

		void Xv(double *v, double *Xv);
		void XTv(double *v, double *XTv);

		void subXv(double *v, double *Xv, int *I, int sizeI);
		void subXTv(double *v, double *XTv, int *I, int sizeI);
};



/**************************************************
Weighting vector class
**************************************************/

class WeightVector
{
	double *w;
	double wscale;
	int n_dim;
	double sq_norm;

	public:
		WeightVector(double *z, int n);
    	void add(const double *x, double c, bool include_bias = false);
		void add(feature_node *x, double c, bool include_bias = false);
		double dot(const double *x);
		double dot(feature_node *x);
		void scale(double c);
		double get_scale();
		void reset_scale();
    	double norm();
		void print();
		double * get_pointer();
		void copy_to(WeightVector &v);
		void set_zero(bool include_bias = false);
		double get_bias();
};


/**************************************************
Loss functions
**************************************************/

class LossFunction
{
	public:
		virtual ~LossFunction() {}

  		virtual double loss(double p, double y) = 0;
		virtual double dloss(double p, double y) = 0;
};

class LogLoss: public LossFunction
{

	public:
		LogLoss() {}
		double loss(double p, double y);
		double dloss(double p, double y);
};

class HingeLoss: public LossFunction
{
	public:
		HingeLoss() {}
		double loss(double a, double y);
		double dloss(double a, double y);
};

class SquaredHingeLoss: public LossFunction
{
	public:
		SquaredHingeLoss() {}
		double loss(double a, double y);
		double dloss(double a, double y);
};

class GaussianLoss: public LossFunction
{
	public:
		GaussianLoss() {}
		double loss(double a, double y);
		double dloss(double a, double y);
};

class PoissonLoss: public LossFunction
{
	public:
		PoissonLoss(double dt=1, bool canonical = true);
		double loss(double a, double y);
		double dloss(double a, double y);

	private:
		double dt;
		bool canonical;
};


/**************************************************************
* Prior classes
**************************************************************/
class Prior
{
	public:
		virtual ~Prior() {}

		virtual std::string get_name() = 0;
		virtual double fun(double *w, int N) = 0;
		virtual void grad(double *w, double *g, int N) = 0;
		virtual void Hv(double *w, double *s, double *Hs, int N) = 0;

		int get_ndim();
		void set_ndim(int d);
		double get_alpha();
		void set_alpha(double a);

	protected:
		double alpha;
		int ndim;
		bool penalize_bias;

		int get_ndim_including_bias(int p);
};


class GaussianPrior : public Prior
{
	public:
		GaussianPrior(double alpha, int ndim = -1, int penalize_bias = -1);
		GaussianPrior(double alpha, double *v, int ndim, int penalize_bias = -1);
		~GaussianPrior();

		std::string get_name() { return "GaussianPrior"; }

		double fun(double *w, int N);
		void grad(double *w, double *g, int N);
		void Hv(double *w, double *s, double *Hs, int N);

	private:
		double *mu;
};


class LaplacePrior : public Prior
{
	public:
		LaplacePrior(double alpha, int ndim = -1, double eps = 1e-6);

		std::string get_name() { return "LaplacePrior"; }

		double fun(double *w, int N);
		void grad(double *w, double *g, int N);
		void Hv(double *w, double *s, double *Hs, int N);

	private:
		double eps;
};


class MixedPrior : public Prior
{
	public:
		MixedPrior(Prior *p1, Prior *p2, double alpha = 1., double gamma = 0.5);

		std::string get_name() { return "MixedPrior"; }

		double fun(double *w, int N);
		void grad(double *w, double *g, int N);
		void Hv(double *w, double *s, double *Hs, int N);

		void set_gamma(double g);
		double get_gamma();

		Prior* get_prior_1() { return this->prior_a; }
		Prior* get_prior_2() { return this->prior_b; }

	protected:
		double gamma;
		Prior *prior_a;
		Prior *prior_b;

		void update_prior_parameters();
};


class ElasticNetPrior : public MixedPrior
{
	public:
		ElasticNetPrior(double alpha, double gamma = 0.5, int ndim = -1, double eps = 1e-12);
		~ElasticNetPrior();

		std::string get_name() { return "ElasticNetPrior"; }

	protected:
		double gamma;
};

class SmoothnessPrior : public Prior
{
	public:
		SmoothnessPrior(double alpha, double gamma, double *S, int ndim);
		~SmoothnessPrior();

		std::string get_name() { return "SmoothnessPrior"; }

		double fun(double *w, int N);
		void grad(double *w, double *g, int N);
		void Hv(double *w, double *s, double *Hs, int N);

		void set_gamma(double g);
		double get_gamma();

	private:
		double gamma;
		double *D;
};


#endif

