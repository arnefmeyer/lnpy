function [kh,logEv,prs, PostCov] = runALDsf(prs0, datastruct, opts)
%
%--------------------------------------------------------------------------
% runALDsf.m: runs Empirical Bayes with ALDsf prior using fmincon
%--------------------------------------------------------------------------
% 
% INPUT ARGUMENTS:
%   prs0 = initial value of hyperparameters of ALDs prior cov. mat.
%   datastruct - has data info such as XX, XY, YY, stimulus dimension
%   opts1 - (optional) options stucture:  'maxiter' and 'tol' etc.
%
% OUTPUT ARGUMENTS:
%   khat - ridge regression RF estimate 
%   logEv - log-evidence at ALDsf solution
%   prs - estimate for hyperparameters
%   PostCov - posterior covariance at ALDsf solution
%
%  (Updated: 23/12/2011 Mijung Park) 

ndims = datastruct.ndims;
leng_ndims = length(ndims);

% set bounds on estimated parameters
noiseRange = [1e-3, 1e3];

% parameters in ALDs covariance
muRange = [zeros(leng_ndims,1), ndims];
gammaRange = [ones(leng_ndims,1), ndims/2];

% parameters in ALDf covariance
m = [-ones(leng_ndims, 1), ones(leng_ndims,1)];
mu = [-ndims, ndims];

% overall 
oscaleRange = [-10, 10];

if leng_ndims ==1 % 1d stimulus
    LBspace = [muRange(:,1); gammaRange(:,1)]; 
    UBspace = [muRange(:,2); gammaRange(:,2)];
    LBfreq = [m(1); mu(1)]; 
    UBfreq = [m(2); mu(2)];
elseif leng_ndims ==2 % 2d stimulus
    phiRange = [-.9.*ones(nchoosek(leng_ndims,2),1), .9.*ones(nchoosek(leng_ndims,2),1)];
    LBspace = [muRange(:,1); gammaRange(:,1); phiRange(:,1)]; 
    UBspace = [muRange(:,2); gammaRange(:,2); phiRange(:,2)];
    LBfreq = [m(:,1); m(1,1); mu(:,1)]; 
    UBfreq = [m(:,2); m(2,2); mu(:,2)];
else % 3d stimulus
    phiRange = [-.2.*ones(nchoosek(leng_ndims,2),1), .2.*ones(nchoosek(leng_ndims,2),1)];
    LBspace = [muRange(:,1); gammaRange(:,1); phiRange(:,1)]; 
    UBspace = [muRange(:,2); gammaRange(:,2); phiRange(:,2)];
    LBfreq = [m(:,1); m(:,1); mu(:,1)]; 
    UBfreq = [m(:,2); m(:,2); mu(:,2)];    
end

LB = [noiseRange(1); LBspace; LBfreq; oscaleRange(1)]; 
UB = [noiseRange(2); UBspace; UBfreq; oscaleRange(2)]; 

lfun= @(p)gradPrior_ALDsf(p, datastruct); 

% ------ Optimize evidence --------------------------------------
prs = fmincon(lfun,prs0,[],[],[],[],LB,UB,[],opts);

% ------ compute filter and posterior variance at maximizer --------------
[logEv,df, kh, PostCov] = lfun(prs);
kh = real(kh); % to remove numerical errors when very small imaginary numbers are included.

fprintf('ALDsf is terminated');