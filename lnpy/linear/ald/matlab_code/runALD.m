function [khatALD, kRidge] = runALD(x, y, spatialdims, nkt)

%--------------------------------------------------------------------------
% runALD.m: find RF estimates using Automatic Locality Determination
%--------------------------------------------------------------------------
%
% DESCRIPTION:
%    Find RF estimates using empirical Bayes with
%    Three localized priors:
%      1. Spacetime localized prior (ALDs),
%      2. Frequency localized prior (ALDf), and
%      3. Spacetime and frequency localized prior (ALDsf)
%
% INPUT ARGUMENTS:
% x        (nt x nx) matrix of design variables (vertical dimension indicates time)
% y        (nt x 1) output variable (column vector)
% ndims   Dimensions of stimuli
%             For example:
%               1d) ndims =[nx];  where nx is either spatial or temporal dimension
%               2d) ndims =[nt; nx],  or [ny; nx]; time by space, or space by space
%               3d) ndims = [nt; ny; nx];  time by space by space
% nkt       number of time samples of x to use to predict y.
%
% OUTPUT ARGUMENTS:
% khatALD           A data structure:
%                   RF estimates / estimated hyperparams / posterior covariance
%
% khatALD.khatS    RF estimate obtained by ALDs
% khatALD.khatF    RF estimate obtained by ALDf
% khatALD.khatSF   RF estimate obtained by ALDsf
%
% khatALD.thetaS   estimated hyperparameters in ALDs
% khatALD.thetaF   estimated hyperparameters in ALDf
% khatALD.thetaSF  estimated hyperparameters in ALDsf
%
% khatALD.postcovS   posterior covariance estimated by ALDs
% khatALD.postcovF   posterior covarinace estimated by ALDf
% khatALD.postcovSF  posterior covariance estimated by ALDsf
%
%
% kRidge               RF estimate by ridge regression
%
% Examples are in testScript.m
%
% (Updated: 25/12/2011 Mijung Park & Jonathan Pillow)

%% Data structure of sufficient statistics from raw data

datastruct = formDataStruct(x, y, nkt, spatialdims);

numb_dims = length(datastruct.ndims);

% ml
% kml = datastruct.xx\datastruct.xy;  

%% Ridge regression for initialization

opts0.maxiter = 1000;  % max number of iterations
opts0.tol = 1e-6;  % stopping tolerance
lam0 = 10;  % Initial ratio of nsevar to prior var (ie, nsevar*alpha)
% ovsc: overall scale, nasevar: noise variance
[kRidge, ovsc ,nsevar]  =  runRidge(lam0, datastruct, opts0);

%% 1. ALDs

% options: ALDs uses trust-region algorithm with analytic gradients and Hessians
% opts1 = optimset('display', 'iter', 'gradobj', 'on', 'hessian', 'on','tolfun',1e-8, 'TolX', 1e-8, 'TolCon', 1e-8, 'MaxIter', 1e3, 'MaxFunEval', 3*1e3);
opts1 = optimset('display', 'iter', 'gradobj', 'on','tolfun',1e-8, 'TolX', 1e-8, 'TolCon', 1e-8, 'MaxIter', 1e3, 'MaxFunEval', 3*1e3);

% Find good initial values
InitialValues = NgridInit_pixel(datastruct.ndims, nsevar, ovsc, kRidge); % make a coarse grid
prs_p = compGriddedLE(@gradLogEv_ALDs, InitialValues, datastruct); % evaluate evidence on the grid

[khatALD.khatS, khatALD.evidS, khatALD.thetaS, khatALD.postcovS] = runALDs(prs_p, datastruct, opts1);

%% 2. ALDf

% ALDf and ALDsf uses active-set algorithm with analytic gradients
opts2 = optimset('display', 'iter', 'gradobj', 'on', 'algorithm','active-set','tolfun',1e-8, 'TolX', 1e-8, 'TolCon', 1e-8, 'MaxIter', 1e3, 'MaxFunEval', 3*1e3);

% Initialize diagonal of M
InitialValues = NgridInit_freq_diag(datastruct.ndims, nsevar, khatALD.thetaS(end)); % make a coarse grid
prs_f = compGriddedLE(@gradPrior_ALDf_diag, InitialValues, datastruct); % evaluate evidence on the grid
% Run ALDf using diagonal M and zero mean to initialize M
[khatALD.khatFdiag, khatALD.evidFdiag, khatALD.thetaFdiag, khatALD.postcovFdiag] = runALDf_diag(prs_f, datastruct, opts2);

mu_init = zeros(numb_dims,1);
if numb_dims==1
    offDiagTrm = [];
else
    offDiagTrm = numb_dims*(numb_dims-1)/2;
end
prsALDf_init = [khatALD.thetaFdiag(1:end-1); 0.1*ones(offDiagTrm,1); mu_init; khatALD.thetaFdiag(end)];

[khatALD.khatF, khatALD.evidF, khatALD.thetaF, khatALD.postcovF] = runALDf(prsALDf_init, datastruct, opts2);

%% 3. ALDsf

% choose min. value among the overall scales of ALDs and ALDf prior cov.
% to optimize from a large Gaussian
ovsc_sf = min(khatALD.thetaS(end), khatALD.thetaF(end));

% set other initial values to those estimated in ALDs and ALDf
prsALDsf_init = [khatALD.thetaS(1:end-1); khatALD.thetaF(2:end-1); abs(ovsc_sf)/10];

[khatALD.khatSF, khatALD.evidSF, khatALD.thetaSF, khatALD.postcovSF] =  runALDsf(prsALDsf_init, datastruct, opts2);

