function [kRidge,alpha,nsevar] = runRidge(lam0, datastruct, opts)

%--------------------------------------------------------------------------
% runRidge.m: find RF estimates using ridge regression
%--------------------------------------------------------------------------
%
% DESCRIPTION:
%      Runs ridge regression, equivalently single-valued ARD  
%      (i.e., computes ML value for the variance of an isotropic Gaussian prior)
%
% INPUT ARGUMENTS:
%   lam0 = initial value of the ratio of noise variance to prior variance
%          (ie. lam0 = nsevar*alpha)
%   nsevar0 - initial value of noise variance
%   x - design matrix 
%   y - spikes
%   opts - (optional) options stucture:  'maxiter' and 'tol'
%
% OUTPUT ARGUMENTS:
%   kRidge - ridge regression estimate of kernel
%   logEv - log-evidence at ARD solution
%   alpha - estimate for inverse prior variance
%   nsevar - estimate for noise variance
%
%  (Updated: 12/2009 JW Pillow) 

MAXALPHA = 1e6; % Maximum allowed value for prior precision

% Check that options field is passed in
if nargin < 2  
    opts.maxiter = 100;
    opts.tol = 1e-6;
    fprintf('Setting options to defaults\n');
end

% ----- Initialize some stuff -------
jcount = 1;  % counter
dparams = inf;  % Change in params from previous step
xx = datastruct.xx;
xy = datastruct.xy;
yy = datastruct.yy;
nstim = datastruct.nstim;

nx= size(xx,1); 
Lmat = eye(nx);  % Diagonal matrix for prior

% ------ Initialize alpha & nsevar using MAP estimate around lam0 ------
kmap0 = (xx + lam0*Lmat)\xy;  
% nsevar = var(y-x*kmap0);
nsevar = yy - 2*kmap0'*xy + kmap0'*xx*kmap0;
alpha = lam0/nsevar;

% ------ Run fixed-point algorithm  ------------
while (jcount <= opts.maxiter) && (dparams>opts.tol) && (alpha <= MAXALPHA)
    CpriorInv = Lmat*alpha;
    [mu,Cprior] = compPostMeanVar([],[],nsevar,CpriorInv,xx,xy);
    alpha2 = (nx- alpha.*trace(Cprior))./sum(mu.^2);
    
% nsevar2 = sum((y-x*mu).^2)./(xlen-sum(1-alpha*diag(Cprior)));
    numerator = yy - 2*mu'*xy + mu'*xx*mu;
    nsevar2 = sum(numerator)./(nstim-sum(1-alpha*diag(Cprior)));

    % update counter, alpha & nsevar
    dparams = norm([alpha2;nsevar2]-[alpha;nsevar]);
    jcount = jcount+1;
    alpha = alpha2;
    nsevar = nsevar2;
end
 
kRidge = (xx + alpha*nsevar*Lmat)\(xy);
% kRidge = compPostMeanVar([],[],nsevar,Lmat*alpha,xx,xy);
% logEv = compLogEv(CpriorInv,nsevar,xx,xy,yy,nstim);

if alpha >= MAXALPHA
    fprintf(1, 'Finished ridge regression: filter is all-zeros\n');
    kRidge = mu*0;  % Prior variance is delta function
elseif jcount < opts.maxiter
    fprintf(1, 'Finished ridge regression in #%d steps\n', jcount)
else
    fprintf(1, 'Finished MAXITER (%d) steps; last step: %f\n', jcount, dparams);
end

