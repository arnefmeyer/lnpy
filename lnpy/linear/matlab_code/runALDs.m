function [khat, logEv, prs, PostCov] = runALDs(prs0, datastruct, opts1)
%
%--------------------------------------------------------------------------
% runALDs.m: runs Empirical Bayes with ALDs prior using fmincon
%--------------------------------------------------------------------------
% 
% INPUT ARGUMENTS:
%   prs0 = initial value of hyperparameters of ALDs prior cov. mat.
%   datastruct - has data info such as XX, XY, YY, stimulus dimension
%   opts1 - (optional) options stucture:  'maxiter' and 'tol' etc.
%
% OUTPUT ARGUMENTS:
%   khat - ridge regression RF estimate 
%   logEv - log-evidence at ALDs solution
%   prs - estimate for hyperparameters
%   PostCov - posterior covariance at ALDs solution
%
%  (Updated: 23/12/2011 Mijung Park) 

ndims = datastruct.ndims;
leng_ndims = length(ndims);

% set bounds on hyperparameters
noiseRange = [1e-3, 1e3];
muRange = [-1+zeros(leng_ndims,1), ndims];
gammaRange = [0.5*ones(leng_ndims,1), ndims/2];
oscaleRange = [-20, 20];

if leng_ndims ==1
    LB = [noiseRange(1); muRange(:,1); gammaRange(:,1);oscaleRange(1)]; 
    UB = [noiseRange(2); muRange(:,2); gammaRange(:,2);oscaleRange(2)];
else 
    phiRange = [-.999.*ones(nchoosek(leng_ndims,2),1), .999.*ones(nchoosek(leng_ndims,2),1)];
    LB = [noiseRange(1); muRange(:,1); gammaRange(:,1); phiRange(:,1);oscaleRange(1)]; 
    UB = [noiseRange(2); muRange(:,2); gammaRange(:,2); phiRange(:,2);oscaleRange(2)]; 
end

lfun = @(p)gradLogEv_ALDs(p, datastruct);

% ------ Optimize evidence --------------------------------------
prs = fmincon(lfun,prs0,[],[],[],[],LB,UB,[],opts1);

% ------ compute filter and posterior variance at maximizer --------------
[logEv,df,ddf,khat,PostCov] = lfun(prs);

fprintf('ALDs is terminated');
