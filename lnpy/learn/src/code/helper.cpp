
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "helper.h"

/* 
	Use CBlas or OpenBlas in weighting vector class?
*/ 
#define CBLAS

#ifdef CBLAS
	#ifdef __cplusplus
		extern "C" {
	#endif

	extern double ddot_(int *n, double *sx, int *incx, double *sy, int *incy);
	extern double dnrm2_(int *n, double *x, int *incx);
	extern int daxpy_(int *n, double *sa, double *sx, int *incx, double *sy, int *incy);
	extern int dscal_(int *n, double *sa, double *sx, int *incx);

	#ifdef __cplusplus
		}
	#endif
#elif defined(OPENBLAS)
	#include <cblas.h>
#endif


/* 
	Define some common wrappers for BLAS/OpenBlas code
*/

double dnrm2_func(int n, double *z, int incr)
{
	#ifdef CBLAS
	return dnrm2_(&n, z, &incr);
	#endif

	#ifdef OPENBLAS
	return cblas_dnrm2(n, z, incr);
	#endif
}

double ddot_func(int n, double *x, int x_incr, double *y, int y_incr)
{
	#ifdef CBLAS
	return ddot_(&n, x, &x_incr, y, &y_incr);
	#endif

	#ifdef OPENBLAS
	return cblas_ddot(n, x, x_incr, y, y_incr);
	#endif
}

int dscal_func(int n, double c, double *x, int incr)
{
	#ifdef CBLAS
	return dscal_(&n, &c, x, &incr);
	#endif

	#ifdef OPENBLAS
	cblas_dscal(n, c, x, incr);
	return 0;
	#endif
}



/**************************************************
Base data class
**************************************************/

void Problem::init_problem(double *Y, int N, int p, double bias, int n_trials=1, double *C=NULL)
{
	this->current_index = 0;
	this->current_pos = -1;
	this->perm_ind = NULL;
	this->perm_len = 0;

	this->N = N;
	this->p = p;
	this->bias = bias;
	this->n_trials = n_trials;
	this->Y = Y;

	if (C != NULL) {
		this->C = C;
		this->delete_C = false;
	} else {
		this->C = new double[N * n_trials];
		for (int i=0; i<N*n_trials; i++)
			this->C[i] = 1;
		this->delete_C = true;
	}
}


void Problem::cleanup_problem()
{
	if (this->delete_C)
		delete[] this->C;

	if (this->perm_ind != NULL)
		delete[] this->perm_ind;
}

double Problem::get_y(int index)
{
	if (index < 0)
		return this->Y[this->current_index];
	else
		return this->Y[index];
}


//void Problem::display()
//{

//	if (this->data_format == SPARSE_FORMAT) {
//		FeatureMatrix X = this->X;

//		int index = 0;
//		for (FeatureMatrix::iterator iter=this->X.begin(); iter != this->X.end(); iter++) {
//			feature_node *s = *iter;
//			printf("index: %d, y = %f, C = %f\n", index, this->Y[index], this->C[index]);
//			while(s->index!=-1)
//			{
//				printf("  X[%d] = %f\n", s->index, s->value);
//				s++;
//			}
//			index++;
//		}
//	} else {
//		// TODO
//	}
//}


void Problem::set_C(double *c)
{
	if (this->delete_C) {
		delete[] this->C;
		this->delete_C = false;
	}
	this->C = c;
}


void Problem::set_permutation(int k, index_t *ind)
{
	this->perm_len = 0;
	this->current_pos = 0;
	this->current_index = 0;

	if (ind != NULL) {
		this->perm_len = k;
		this->perm_ind = new index_t[k];
		memcpy(this->perm_ind, ind, k * sizeof(index_t));
		this->current_index = ind[0];
	}
}


int Problem::next()
{
	int status = 0;

	this->current_pos += 1;

	if (this->perm_len == 0 || this->perm_ind == NULL) {
		if (this->current_pos >= this->get_N())
			this->current_pos = 0;
		this->current_index = this->current_pos;
	}
	else {

		if (this->current_pos >= this->perm_len)
			status = -1;
		else
			this->current_index = this->perm_ind[this->current_pos];
	}

	return status;
}


index_t Problem::get_current_index()
{
	std::cout << this->current_index << std::endl;
	return this->current_index;
}


void Problem::reset_index()
{
	this->current_pos = 0;
	this->current_index = 0;
}


//double Problem::xv(double *v)
//{
//	double z = 0;
//	index_t index;
//	index = this->get_current_index();

//	if (this->data_format == SPARSE_FORMAT) {
//		FeatureMatrix X = this->X_sparse;
//		while(s->index!=-1)
//		{
//			z += v[s->index] * s->value;
//			s++;
//		}
//	}
//	else {
//		double *X = this->X_dense;
//		int ndim = this->p;
//		int n_trials = this->n_trials;
//		int incr = 1;
//		double *x = &X[index*ndim];
//		z = ddot_func(ndim, v, incr, x, incr) + bias*v[ndim];
//	}

//	return z;
//}



/**************************************************
Sparse data class adapted from liblinear
**************************************************/

SparseProblem::SparseProblem(double *X, double *Y, int N, int p, double bias, int n_trials=1, double *C=NULL)
{
	init_problem(Y, N, p, bias, n_trials, C);
	repack_data_sparse(X, N, p, bias, n_trials);
}


SparseProblem::~SparseProblem()
{
	int count = 0;

	for (FeatureMatrix::iterator i=this->X.begin(); i != this->X.end(); i++) {
		delete[] *i;
		(*i) = NULL;
		count++;
	}

	cleanup_problem();
}


void SparseProblem::repack_data_sparse(double *X, int N, int p, double bias, int n_trials)
{
	FeatureMatrix data;
	bool fit_bias = (bias > 0);
	int n_dim = p + (int)fit_bias;

	for (int i=0; i<N; i++) {
		
		struct feature_node *tmp = new feature_node[n_dim+1];
		for (int j=0; j<p; j++) {
			tmp[j].index = j;
			tmp[j].value = X[i*p+j];
		}
		
		if (fit_bias) {
			tmp[n_dim-1].index = p;
			tmp[n_dim-1].value = bias;
		}
		tmp[n_dim].index = -1;
		tmp[n_dim].value = -1;

		data.push_back(tmp);
	}

	this->X = data;
}

double SparseProblem::xv(double *v)
{
	double z = 0;
	index_t index;
	index = this->get_current_index();

	FeatureMatrix X = this->X;
	feature_node *s = X.at(index);
	while(s->index!=-1)
	{
		z += v[s->index] * s->value;
		s++;
	}

	return z;
}

void SparseProblem::Xv(double *v, double *Xv)
{
	int N = this->get_N();
	int n_trials = this->get_ntrials();

	FeatureMatrix X = this->X;

	int idx = 0;
	double z;

	for(int i=0;i<N;i++)
	{
		feature_node *s = X[i];
		z = 0;
		while(s->index!=-1)
		{
			z += v[s->index] * s->value;
			s++;
		}

		for (int j=0; j<n_trials; j++)
			Xv[idx++] = z;
	}
}

void SparseProblem::XTv(double *v, double *XTv)
{
	int N = this->get_N();
	int w_size = this->get_ndim() + (int)this->get_fit_bias();
	FeatureMatrix X = this->X;

	for(int i=0;i<w_size;i++)
		XTv[i] = 0;

	for(int i=0;i<N;i++)
	{
		feature_node *s = X[i];
		while(s->index!=-1)
		{
			XTv[s->index] += v[i] * s->value;
			s++;
		}
	}
}

void SparseProblem::subXv(double *v, double *Xv, int *I, int sizeI)
{
	FeatureMatrix X = this->X;;

	for(int i=0;i<sizeI;i++)
	{
		feature_node *s = X[I[i]];
		Xv[i] = 0;
		while(s->index!=-1)
		{
			Xv[i] += v[s->index] * s->value;
			s++;
		}
	}
}

void SparseProblem::subXTv(double *v, double *XTv, int *I, int sizeI)
{
	int w_size = this->get_ndim() + (int)this->get_fit_bias();
	FeatureMatrix X = this->X;
	int idx;

	for(int i=0;i<w_size;i++)
		XTv[i] = 0;

	for(int i=0;i<sizeI;i++)
	{
		feature_node *s = X[I[i]];
		while(s->index!=-1)
		{
			XTv[s->index] += v[i] * s->value;
			s++;
		}
	}
}

//feature_node* Problem::get_x(int index)
//{
//	if (index < 0)
//		return this->X.at(this->current_index);
//	else
//		return this->X.at(index);
//}

void* SparseProblem::get_x(int index)
{
	int idx = index;
	if (index < 0)
		idx = this->current_index;

	return (void*)this->X.at(idx);
}



/**************************************************
Class for dense data
**************************************************/


DenseProblem::DenseProblem(double *X, double *Y, int N, int p, double bias, int n_trials=1, double *C=NULL)
{
	init_problem(Y, N, p, bias, n_trials, C);
	this->X = X;
}


DenseProblem::~DenseProblem()
{
	cleanup_problem();
}

void* DenseProblem::get_x(int index)
{
	int ndim = this->p;
	int idx = index;
	if (index < 0)
		idx = this->current_index;

	return (void*)&this->X[idx * ndim];
}


double DenseProblem::xv(double *v)
{
	double z = 0;
	index_t index;
	index = this->get_current_index();

	double *X = this->X;
	int ndim = this->p;
	int incr = 1;
	double *x = &X[index*ndim];
	z = ddot_func(ndim, v, incr, x, incr) + bias*v[ndim];

	return z;
}


void DenseProblem::Xv(double *v, double *Xv)
{
	int N = this->get_N();
	int n_trials = this->n_trials;
	double *X = this->X;
	bool fit_bias = this->get_fit_bias();
	double b = this->bias;
	
	int idx = 0;
	int incr = 1;
	int ndim = this->p;
	double z;

	for (int i=0; i<N; i++)
	{
		double *x = &X[i*ndim];
		z = ddot_func(ndim, v, incr, x, incr);
		if (fit_bias)
			z += b*v[ndim];

		// The values on repeated trials will be identical
		for (int j=0; j<n_trials; j++)
			Xv[idx++] = z;
	}
}

void DenseProblem::XTv(double *v, double *XTv)
{
	int N = this->get_N();
	int n_trials = this->n_trials;
	double *X = this->X;

	int ndim = this->p;
	bool fit_bias = this->get_fit_bias();
	double b = this->bias;

	for(int i=0;i<ndim+(int)fit_bias;i++)
		XTv[i] = 0;

	for (int i=0; i<N; i++) {
		for (int j=0; j<ndim; j++) {
			XTv[j] += X[i*ndim+j] * v[i] * n_trials;
		}
	}

	if (fit_bias) {
		for (int i=0; i<N; i++) {
			XTv[ndim] += v[i] * b * n_trials;
		}
	}
}

void DenseProblem::subXv(double *v, double *Xv, int *I, int sizeI)
{
	double *X = this->X;
	bool fit_bias = this->get_fit_bias();
	double b = this->bias;
	
	int idx = 0;
	int incr = 1;
	int ndim = this->p;
	double z;

	for (int i=0; i<sizeI; i++)
	{
		idx = I[i] / n_trials;
		double *x = &X[idx*ndim];
		z = ddot_func(ndim, v, incr, x, incr);
		if (fit_bias)
			z += b*v[ndim];

		// The values on repeated trials will be identical
//		for (int j=0; j<n_trials; j++)
		Xv[i] = z;
	}
}

void DenseProblem::subXTv(double *v, double *XTv, int *I, int sizeI)
{
	double *X = this->X;

	int ndim = this->p;
	bool fit_bias = this->get_fit_bias();
	double b = this->bias;
	int idx = 0;

	for(int i=0;i<ndim+(int)fit_bias;i++)
		XTv[i] = 0;

	for (int i=0; i<sizeI; i++) {
		idx = I[i] / n_trials;
		for (int j=0; j<ndim; j++) {
			XTv[j] += X[idx*ndim+j] * v[i];
		}
	}

	if (this->get_fit_bias()) {
		for (int i=0; i<sizeI; i++) {
			XTv[ndim] += v[i] * b;
		}
	}
}


/**************************************************
Weighting vector class
**************************************************/

WeightVector::WeightVector(double *v, int n)
{
	#ifdef OPENBLAS
	openblas_set_num_threads(1);
	goto_set_num_threads(1);
	#endif

	this->w = v;
	this->wscale = 1.;
	this->n_dim = n;

	int incr = 1;
	double norm = dnrm2_func(n, v, incr);
	this->sq_norm = norm * norm;
}


void WeightVector::add(const double *x, double c, bool include_bias)
{
	double x_sq_norm;
	double innerprod;
	int p;

	x_sq_norm = 0;
	innerprod = 0;
	p = this->n_dim;

	for (int i=0; i<p; i++) {
		innerprod += (this->w[i] * x[i]);
		x_sq_norm += (x[i] * x[i]);
		this->w[i] += x[i] * (c / this->wscale);
	}

	if (include_bias)
		this->w[p] += c * x[p];

	this->sq_norm += (x_sq_norm * c * c) + (2.0 * innerprod * this->wscale * c);
}


void WeightVector::add(feature_node *x, double c, bool include_bias)
{
	double x_sq_norm;
	double innerprod;

	x_sq_norm = 0;
	innerprod = 0;
	while (x->index != -1) {
		innerprod += (this->w[x->index] * x->value);
		x_sq_norm += (x->value * x->value);
		this->w[x->index] += x->value * (c / this->wscale);
		x++;
	}

	
	if (include_bias) {
		x--;
		innerprod -= (this->w[x->index] * x->value);
		x_sq_norm -= (x->value * x->value);
		this->w[x->index] += x->value * (1 - c / this->wscale);
		x++;
	}

	this->sq_norm += (x_sq_norm * c * c) + (2.0 * innerprod * this->wscale * c);
}


double WeightVector::dot(const double *x)
{
	double z;

	int incr = 1;
	z = ddot_func(this->n_dim, this->w, incr, const_cast<double*>(x), incr);
	z *= this->wscale;

	return z;
}

double WeightVector::dot(feature_node *x)
{
	double z = 0;

	while(x->index!=-1)
	{
		z += this->w[x->index] * x->value;
		x++;
	}

	z *= this->wscale;
	return z;
}

void WeightVector::scale(double c)
{
	this->wscale *= c;
	this->sq_norm *= (c * c);
	if (this->wscale < 1e-10 || this->wscale > 1e10)
   		reset_scale();
}

double WeightVector::get_scale()
{
	return this->wscale;
}

void WeightVector::reset_scale()
{
	int incr = 1;
	dscal_func(this->n_dim, this->wscale, this->w, incr);

	this->wscale = 1;
}

double WeightVector::norm()
{
	return sqrt(this->sq_norm);
}

void WeightVector::print()
{
	for (int i=0; i<this->n_dim; i++)
		printf("%0.3f ", this->w[i]);
	printf("\n");
}

double * WeightVector::get_pointer()
{
	return this->w;
}

void WeightVector::copy_to(WeightVector &v)
{
	for (int i=0; i<this->n_dim; i++)
		v.w[i] = this->w[i];
	v.wscale = this->wscale;
	v.sq_norm = this->sq_norm;
}

void WeightVector::set_zero(bool include_bias)
{
	int p = this->n_dim + (int)include_bias;
	for (int i=0; i<p; i++)
		this->w[i] = 0;

	this->sq_norm = 0;
	this->wscale = 1;
}

double WeightVector::get_bias()
{
	return this->w[this->n_dim];
}




/**************************************************
Loss functions (for SGD-based optimization)
**************************************************/

// logloss(a,y) = log(1+exp(-a*y))
double LogLoss::loss(double p, double y)
{
	double z = p * y;
	if (z > 18) 
	  return exp(-z);
	if (z < -18)
	  return -z;
	return log(1 + exp(-z));
}

// -dloss(a,y)/da
double LogLoss::dloss(double p, double y)
{
	double z = p * y;
	if (z > 18) 
	  return y * exp(-z);
	if (z < -18)
	  return y;
	return y / (1 + exp(z));
}


// hingeloss(a,y) = max(0, 1-a*y)
double HingeLoss::loss(double a, double y)
{
	double z = a * y;
	if (z > 1) 
		return 0;
	return 1 - z;
}

// -dloss(a,y)/da
double HingeLoss::dloss(double a, double y)
{
	double z = a * y;
	if (z > 1) 
		return 0;
	return y;
}


// squaredhingeloss(a,y) = 1/2 * max(0, 1-a*y)^2
double SquaredHingeLoss::loss(double a, double y)
{
	double z = a * y;
	if (z > 1)
	  return 0;
	double d = 1 - z;
	return 0.5 * d * d;
}

// -dloss(a,y)/da
double SquaredHingeLoss::dloss(double a, double y)
{
	double z = a * y;
	if (z > 1) 
	  return 0;
	return y * (1 - z);
}

// gaussianloss(a,y) = 1/2 * (a - y)^2
double GaussianLoss::loss(double a, double y)
{
	double d = a - y;
	return 0.5 * d * d;
}

// -dloss(a,y)/da
double GaussianLoss::dloss(double a, double y)
{
	return y - a;
}

PoissonLoss::PoissonLoss(double dt, bool canonical)
{
	this->dt = dt;
	this->canonical = canonical;
}

double PoissonLoss::loss(double z, double y)
{
	double d;
	if (canonical || z <= 0) {
		if (z < -120)
			z = -120;
		if (z > 50)
			z = 50;
		d = y * z - exp(z);
	} else {
		double u = 1 + z + .5*z*z;
		d = y * log(u) - u;
	}

//	std::cout << z << " | " << d << std::endl;
	return d;
}

double PoissonLoss::dloss(double z, double y)
{
	double d;
	if (canonical || z <= 0) {
		if (z < -120)
			z = -120;
		if (z > 50)
			z = 50;
		d = y - exp(z);
	} else {
		d = (1 + z) * (y / (1 + z + .5*z*z) - dt);
	}
	return d;
}


/**************************************************
Priors
**************************************************/

/*
	Prior base class
*/
double Prior::get_alpha()
{
	return this->alpha;
}

void Prior::set_alpha(double a)
{
	this->alpha = a;
}

int Prior::get_ndim()
{
	return this->ndim;
}

void Prior::set_ndim(int d)
{
	this->ndim = d;
}

int Prior::get_ndim_including_bias(int p)
{
	int d;
	if (this->ndim > 0)
		d = this->ndim + (int)(this->penalize_bias > 0);
	else
		d = p + (int)(this->penalize_bias > 0);

	return d;
}


/*
	Gaussian prior (L2-norm regularizer) with mean
	vector
*/
GaussianPrior::GaussianPrior(double alpha, int ndim, int penalize_bias)
{
	this->alpha = alpha;
	this->ndim = ndim;
	this->penalize_bias = penalize_bias;
	this->mu = NULL;
}

GaussianPrior::GaussianPrior(double alpha, double *mean, int ndim, int penalize_bias)
{
	this->alpha = alpha;
	this->ndim = ndim;
	this->penalize_bias = penalize_bias;

	int d = get_ndim_including_bias(ndim);
	this->mu = new double[d];
	memcpy(this->mu, mean, d * sizeof(double));
}

GaussianPrior::~GaussianPrior()
{
	if (this->mu != NULL)
		delete[] this->mu;
}

double GaussianPrior::fun(double *w, int p)
{
	double alpha = this->alpha;
	double f = 0;
	int d = get_ndim_including_bias(p);
	double u;

	if (this->mu != NULL)
		for (int i=0; i<d; i++) {
			u = w[i] - this->mu[i];
			f += .5 * alpha * u * u;
		}
	else
		for (int i=0; i<d; i++)
			f += .5 * alpha * w[i] * w[i];

	return f;
}

void GaussianPrior::grad(double *w, double *g, int p)
{
	double alpha = this->alpha;
	int d = get_ndim_including_bias(p);
	double u;

	if (this->mu != NULL)
		for (int i=0; i<d; i++) {
			u = w[i] - this->mu[i];
			g[i] += alpha * u;
		}
	else
		for (int i=0; i<d; i++)
			g[i] += alpha * w[i];
}

void GaussianPrior::Hv(double *w, double *s, double *Hs, int p)
{
	double alpha = this->alpha;
	double d = get_ndim_including_bias(p);

	for (int i=0; i<d; i++) {
		Hs[i] += alpha * s[i];
	}
}


/*
	Laplace prior (L1-norm regularizer)
*/
LaplacePrior::LaplacePrior(double alpha, int ndim, double eps)
{
	this->alpha = alpha;
	this->ndim = ndim;
	this->eps = eps;
}

double LaplacePrior::fun(double *w, int p)
{
	double alpha = this->alpha;
	double eps = this->eps;
	double f = 0;
	int d = get_ndim_including_bias(p);

	for (int i=0; i<d; i++)
		f += alpha * sqrt(w[i] * w[i] + eps);

	return f;
}

void LaplacePrior::grad(double *w, double *g, int p)
{
	double alpha = this->alpha;
	double eps = this->eps;
	int d = get_ndim_including_bias(p);

	for (int i=0; i<d; i++)
		g[i] += alpha * (w[i] / (sqrt(w[i]*w[i] + eps)));
}

void LaplacePrior::Hv(double *w, double *s, double *Hs, int p)
{
	double alpha = this->alpha;
	double eps = this->eps;
	int d = get_ndim_including_bias(p);

	for (int i=0; i<d; i++)
		Hs[i] += alpha * (pow(w[i]*w[i] + eps, -.5) - w[i]*w[i] * pow(w[i]*w[i] + eps, -1.5)) * s[i];
}


/*
	Mixed prior (mixture of two regularizers)
*/
MixedPrior::MixedPrior(Prior *p, Prior *q, double alpha, double gamma)
{
	this->alpha = alpha;
	this->gamma = gamma;
	this->prior_a = p;
	this->prior_b = q;
}

double MixedPrior::fun(double *w, int p)
{
	double f = 0;
	update_prior_parameters();
	f += this->prior_a->fun(w, p);
	f += this->prior_b->fun(w, p);

	return f;
}

void MixedPrior::grad(double *w, double *g, int p)
{
	update_prior_parameters();
	this->prior_a->grad(w, g, p);
	this->prior_b->grad(w, g, p);
}

void MixedPrior::Hv(double *w, double *s, double *Hs, int p)
{
	update_prior_parameters();
	this->prior_a->Hv(w, s, Hs, p);
	this->prior_b->Hv(w, s, Hs, p);
}

void MixedPrior::set_gamma(double g)
{
	this->gamma = g;
}

double MixedPrior::get_gamma()
{
	return this->gamma;
}

void MixedPrior::update_prior_parameters()
{
	double alpha = this->alpha;
	double gamma = this->gamma;

	this->prior_a->set_alpha(alpha * gamma);
	this->prior_b->set_alpha(alpha * (1 - gamma));
}


/*
	Elastic net prior (mixture of L1-norm and L2-norm regularizer)
*/
ElasticNetPrior::ElasticNetPrior(double alpha, double gamma, int ndim, double eps)
: MixedPrior(NULL, NULL, alpha, gamma)
{
	this->prior_a = new GaussianPrior(1., ndim);
	this->prior_b = new LaplacePrior(1., ndim, eps);
}

ElasticNetPrior::~ElasticNetPrior()
{
	delete this->prior_a;
	delete this->prior_b;
}


/*
	Smoothness prior
*/

SmoothnessPrior::SmoothnessPrior(double alpha, double gamma, double *S, int n_dim)
{
	this->alpha = alpha;
	this->gamma = gamma;
	this->D = new double[n_dim * n_dim];
	this->ndim = n_dim;
	memcpy(this->D, S, n_dim * n_dim * sizeof(double));
}

SmoothnessPrior::~SmoothnessPrior()
{
	if (this->D != NULL)
		delete[] this->D;
}

double SmoothnessPrior::fun(double *w, int p)
{
	double alpha = this->alpha;
	double gamma = this->gamma;
	double *S = this->D;
	int d = get_ndim_including_bias(p);
	double f = 0;

	for (int i=0; i<d; i++) {
		f += alpha * w[i] * w[i];
		for (int j=0; j<d; j++)
			f += gamma * w[j] * S[i*p + j] * w[i];
	}

	f /= 2.;

	return f;
}

void SmoothnessPrior::grad(double *w, double *g, int p)
{
	double alpha = this->alpha;
	double gamma = this->gamma;
	double *S = this->D;
	int d = get_ndim_including_bias(p);

	for (int i=0; i<d; i++) {
		g[i] += alpha * w[i];
		for (int j=0; j<d; j++)
			g[i] += gamma * w[i] * S[i*p + j];
	}
}

void SmoothnessPrior::Hv(double *w, double *s, double *Hs, int p)
{
	double alpha = this->alpha;
	double gamma = this->gamma;
	double *S = this->D;
	int d = get_ndim_including_bias(p);

	for (int i=0; i<d; i++) {
		Hs[i] += alpha * s[i];
		for (int j=0; j<d; j++)
			Hs[i] += gamma * S[i*p + j] * s[i];
	}
}

void SmoothnessPrior::set_gamma(double g)
{
	this->gamma = g;
}

double SmoothnessPrior::get_gamma()
{
	return this->gamma;
}

