
#include "helper.h"


class SGD
{
	public:
		SGD(double *v, int n_dim, double fit_bias, LossFunction *loss = new GaussianLoss(), double n_epochs = 1, double alpha = 1., bool warm_start = false);
		~SGD();
		int fit(Problem *prob, int start_iter = 0, int verbose = 0);
		double* get_weights();
		double get_bias();
		int get_ndims() { return this->n_dim; }
		bool get_fit_bias() { return this->fit_bias; }

		void set_alpha(double a);
		double get_alpha();

	protected:
		WeightVector *w;
		int n_dim;
		bool fit_bias;
		LossFunction *loss_fun;
		double n_epochs;
		double alpha;
		bool warm_start;	
};


class ASGD : public SGD
{
	public:
		ASGD(double *v, double *va, int n_dim, double fit_bias, LossFunction *loss = new GaussianLoss(), double n_epochs = 1, double alpha = 1., bool warm_start = false, int start_avg = 0, double decay_rate = 2.);
		~ASGD();
		int fit(Problem *prob, int start_iter = 0, int verbose = 0);
		double* get_weights();
		double get_bias();

	protected:
		WeightVector *wa;
		int start_averaging;
		double poly_decay;
};

