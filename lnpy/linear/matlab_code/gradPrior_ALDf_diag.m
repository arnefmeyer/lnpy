function [f, df, khat, LL] = gradPrior_ALDf_diag(prs, datastruct)
%
%--------------------------------------------------------------------------
% gradPrior_ALDf_diag: compute gradients of ALDf_diag prior covariance matrix
%--------------------------------------------------------------------------
% 
% INPUT ARGUMENTS:
%   prs - initial value of hyperparameters of ALDf prior cov. mat.
%   datastruct - has data info such as XX, XY, YY, stimulus dimension
%
% OUTPUT ARGUMENTS:
%   f - maximum evidence given data
%   df - df/dtheta at f = f_max
%   khat - estimate of kernel
%   LL - posterior covariance at f = f_max
%
%  (Updated: 23/12/2011 Mijung Park) 

%  Unpack the data information:
XX = datastruct.xxf;
XY = datastruct.xyf;
YY = datastruct.yy;
nsamps = datastruct.nstim;
ndims = datastruct.ndims;

% Unpack params
nsevar = abs(prs(1));
prs = prs(2:end); % Hyper-params, except noise variance

% Check total dimension of the filter
leng_ndims = length(ndims);

%% form Cprior based on coordinate dimension

switch leng_ndims
    case (1)
        
        nx = ndims(1);
        
        M = prs(1); % unpack params
        ovsc = prs(2);
        
        fx = FFTaxis(nx); % coordinates in Fourier domain
        w = fx(:);
        
    case (2)
        
        ny = ndims(1);
        nx = ndims(2);
        
        M = [prs(1) 0; 0 prs(2)]; % unpack params
        ovsc = prs(3);
        
        [fy, fx] =  FFT2axis(ny,nx);  % coordinates in Fourier domain
        w = [fy(:) fx(:)];
        
    case (3)
        
        nt = ndims(1);
        ny = ndims(2);
        nx = ndims(3);
        
        M = [prs(1) 0 0;0 prs(2) 0;0 0 prs(3)]; % unpack params
        ovsc = prs(4);        
        
        [ft, fy, fx] =  FFT3axis(nt, ny, nx); % coordinates in Fourier domain
        w = [ft(:) fy(:) fx(:)];
        
    otherwise
        
        disp('this dimension is not applicable');
        
end

X = (M*w')';
cdiag = exp(-.5*sum(X.*X,2)- ovsc); % diag of ALDf prior cov. mat.

svMin = 1e-6; % singular value threshold for eliminating dimensions  to make code fast
svthresh = max(cdiag)*svMin;

% prune data based on frequency sparsity
if min(cdiag)>svthresh
    iikeep = true(length(cdiag), 1);
else
    iikeep = (cdiag>=svthresh);
    X = X(iikeep, :);
    w = w(iikeep, :);
    cdiag = cdiag(iikeep);
end

Cprior = diag(cdiag);

%% evaluate log-evidence & compute 1st/2nd Derivatives w.r.t. noise variance

[f, khat, LL, df1, ddf1, LLinvC] = gradLogEv(Cprior, abs(nsevar),XX(iikeep, iikeep),XY(iikeep),YY,nsamps);
f = -f;

%% Gradients of Cprior w.r.t each hyperparam

numb_params = length(prs); % alpha, mu, gam, phi
leng_keep = sum(iikeep);
I = ones(leng_keep,1);

cGrad = zeros(leng_keep, numb_params); % cGrad = dC/dtheta   I = ones(leng_keep,1);

% dM/dm_i
der_Mw = zeros(leng_keep, leng_ndims, leng_ndims);

if leng_ndims==1
    der_Mw(:,:,1) = I;
else
    if leng_ndims == 2
        der_Mw(:,1,1) = I;
        der_Mw(:,2,2) = I;
    else %leng_ndims ==3
        der_Mw(:,1,1) = I;
        der_Mw(:,2,2) = I;
        der_Mw(:,3,3) = I;
    end
end

% gradients wrt M
for i=1:numb_params-1
    cGrad(:,i) = - sum(X.*(der_Mw(:,:,i).*w),2);
end

% gradients wrt overall scale
cGrad(:,end) = - I;

%% Gradients of evidence w.r.t. hyperparams

diagTrm = I - diag(LLinvC) - sum((khat*(XY(iikeep))').*transpose(LLinvC), 2)./nsevar;
dfdthet = - .5*cGrad'*diagTrm;  % 0.5* Tr(C - Lambda - k*k')*dCinv/dtheta);
df = -[df1; dfdthet];
df = real(df);

%% project back to the original space

nx = size(XX,1);

khatreduced = khat; % khat (posterior mean)
khat = zeros(nx, 1);
khat(iikeep) = khatreduced;

Lmat = zeros(nx, nx); % LL (posterior covariance)
Lmat(iikeep, iikeep) = LL;
LL = Lmat;


