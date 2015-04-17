
#include "sgd.h"


/***********************************************

Plain stochastic gradient descent (SGD) solver
using learning rate decay (aka annealing schedule)

************************************************/

SGD::SGD(double *v, int n_dim, double fit_bias, LossFunction *loss, double n_epochs, double alpha, bool warm_start)
{
	this->w = new WeightVector(v, n_dim);
	this->n_dim = n_dim;
	this->fit_bias = fit_bias > 0;
	this->loss_fun = loss;
	this->n_epochs = n_epochs;
	this->alpha = alpha;
	this->warm_start = warm_start;
}

SGD::~SGD()
{
	delete this->w;
}

int SGD::fit(Problem *prob, int start_iter, int verbose)
{
	int iter;
	int max_iter;
	double lambda;
	LossFunction *loss;

	WeightVector *v;
	void *x;
	bool is_sparse;
	double y;
	int t;
	double eta;		
	double z;
	double update;

	iter = 0;
	max_iter = start_iter + (int)(this->n_epochs * prob->get_N());
	lambda = this->alpha;
	loss = this->loss_fun;

	v = this->w;
	if (!this->warm_start)
		v->set_zero(this->fit_bias);

	is_sparse = prob->is_sparse();

	while (1) {

		/* Pick random example (permutation provided by Problem object) */
		x = prob->get_x();
		y = prob->get_y();

		/* Learning rate (aka annealing schedule) */
		t = start_iter + iter;
		eta = 1. / (lambda * t + 1.);

        /* Regularizer step (don't penalize bias term!) */
		v->scale(1. - eta * lambda);

        /* Loss step */
		if (is_sparse)
			z = v->dot((feature_node*)x);
		else
			z = v->dot((double*)x);
		update = loss->dloss(z, y);
		if (update != 0) {
			if (is_sparse)
				v->add((feature_node*)x, eta * update, this->fit_bias);
			else
				v->add((double*)x, eta * update, this->fit_bias);
		}

		/* Check termination criterion */
		if (prob->next() < 0 || ++iter >= max_iter)
			break;
	}

	v->reset_scale();

	return iter;
}

double* SGD::get_weights()
{
	return this->w->get_pointer();
}

double SGD::get_bias()
{
	if (this->fit_bias)
		return this->w->get_bias();
	else
		return 0;
}

void SGD::set_alpha(double a)
{
	this->alpha = a;
}

double SGD::get_alpha()
{
	return this->alpha;
}



/***********************************************

Polynomial-averaging SGD (Shamir et al. ICML 2012)

************************************************/

ASGD::ASGD(double *v, double *va, int n_dim, double fit_bias, LossFunction *loss, double n_epochs, double alpha, bool warm_start, int start_avg, double avg_decay) : SGD(v, n_dim, fit_bias, loss, n_epochs, alpha, warm_start)
{
	this->wa = new WeightVector(va, n_dim);
	this->start_averaging = start_avg;
	this->poly_decay = avg_decay;
}

ASGD::~ASGD()
{
	delete this->wa;
}

int ASGD::fit(Problem *prob, int start_iter, int verbose)
{
	int iter;
	int max_iter;
	double lambda;
	LossFunction *loss;
	int start_avg;
	double avg_decay;
	int ndim;

	WeightVector *v;
	WeightVector *va;
	double *v_ptr;
	double *va_ptr;
	bool is_sparse;
	void *x;
	double y;
	int t;
	double eta;		
	double z;
	double update;
	double avg_update;

	iter = 0;
	max_iter = start_iter + (int)(this->n_epochs * prob->get_N());
	lambda = this->alpha;
	loss = this->loss_fun;
	start_avg = this->start_averaging;
	avg_decay = this->poly_decay;
	ndim = this->n_dim;

	v = this->w;
	va = this->wa;
	v_ptr = v->get_pointer();
	va_ptr = va->get_pointer();
	if (!this->warm_start) {
		v->set_zero(this->fit_bias);
		va->set_zero(this->fit_bias);
	}

	is_sparse = prob->is_sparse();

	while (1) {

		/* Pick random example (permutation provided by Problem object) */
		x = prob->get_x();
		y = prob->get_y();

		/* Learning rate (aka annealing schedule) */
		t = start_iter + iter;
		eta = 1. / (lambda * t + 1.);

        /* Regularizer step (don't penalize bias term!) */
		v->scale(1. - eta * lambda);

        /* Loss step */
		if (is_sparse)
			z = v->dot((feature_node*)x);
		else
			z = v->dot((double*)x);
		update = loss->dloss(z, y);
		if (update != 0) {
			if (is_sparse)
				v->add((feature_node*)x, eta * update, this->fit_bias);
			else
				v->add((double*)x, eta * update, this->fit_bias);
		}

		if (iter >= start_avg) {
			avg_update = (avg_decay + 1.)/(t + avg_decay);
			va->scale(1. - avg_update);
			va->add(v->get_pointer(), v->get_scale() * avg_update, false);
			if (this->fit_bias)
				va_ptr[ndim] = (1. - avg_update) * va_ptr[ndim] + avg_update * v_ptr[ndim];
		}

//		std::cout << "iter: " << iter << " | eta: " << eta << " | update: " << update << " | w: " << v->norm() << " | wa: " << va->norm() << std::endl;

		/* Check termination criterion */
		if (prob->next() < 0 || ++iter >= max_iter)
			break;
	}

	v->reset_scale();
	va->reset_scale();

	return iter;
}

double* ASGD::get_weights()
{
	return this->wa->get_pointer();
}

double ASGD::get_bias()
{
	if (this->fit_bias)
		return this->wa->get_bias();
	else
		return 0;
}

