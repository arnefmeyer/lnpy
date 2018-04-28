function [khat,logEv, prs, PostCov] = runALDf_diag(prs0, datastruct, opts)
%
%--------------------------------------------------------------------------
% runALDf_diag.m: runs Empirical Bayes with ALDf_diag prior (*M is diagonal and zero mean)
%--------------------------------------------------------------------------
% 
% INPUT ARGUMENTS:
%   prs0 = initial value of hyperparameters of ALDf prior cov. mat.
%   datastruct - has data info such as XX, XY, YY, stimulus dimension
%   opts1 - (optional) options stucture:  'maxiter' and 'tol' etc.
%
% OUTPUT ARGUMENTS:
%   khat - ridge regression RF estimate 
%   logEv - log-evidence at ALDf_diag solution
%   prs - estimate for hyperparameters
%   PostCov - posterior covariance at ALDf_diag solution
%
%  (Updated: 23/12/2011 Mijung Park) 

ndims = datastruct.ndims;
leng_ndims = length(ndims);

% set bounds on estimated parameters
noiseRange = [1e-3, 1e3];
m = [-ones(leng_ndims, 1), ones(leng_ndims,1)];
oscaleRange = [-20, 20];

if leng_ndims ==1
    LB = [noiseRange(1); m(1); oscaleRange(1)]; 
    UB = [noiseRange(2); m(2); oscaleRange(2)];
elseif leng_ndims ==2
    LB = [noiseRange(1); m(:,1); oscaleRange(1)]; 
    UB = [noiseRange(2); m(:,2); oscaleRange(2)];
else % 3d
    LB = [noiseRange(1); m(:,1); oscaleRange(1)]; 
    UB = [noiseRange(2); m(:,2); oscaleRange(2)];    
end

lfun = @(p)gradPrior_ALDf_diag(p, datastruct);

% ------ Optimize evidence --------------------------------------
prs = fmincon(lfun,prs0,[],[],[],[],LB,UB,[],opts);

% ------ compute filter and posterior variance at maximizer --------------
[logEv,df, kh, PostCov] = lfun(prs);

if leng_ndims==1
    M = FFTmatrix(ndims);
    khat_cc = M'*kh;
elseif leng_ndims==2
    nt = ndims(1);
    nx = ndims(2);
    M = FFT2matrix(nt, nx);
    khat_cc = reshape(M'*kh, nt, nx);
else % if k is 3 dim.
    nt = ndims(1);
    ny = ndims(2);
    nx = ndims(3);
    BB = FFT3matrix(nt, ny, nx);
    khat_cc = permute(reshape(BB'*kh(:), nt, ny, nx), [2 3 1]);
end

khat = real(khat_cc); % to remove numerical errors (small imaginary parts)

fprintf('ALDf (diag) is terminated');