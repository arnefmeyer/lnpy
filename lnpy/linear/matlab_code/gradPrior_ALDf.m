function [f, df, khat, LL] = gradPrior_ALDf(prs, datastruct)
%
%--------------------------------------------------------------------------
% gradPrior_ALDf: compute gradients of ALDs prior covariance matrix
%--------------------------------------------------------------------------
% 
% INPUT ARGUMENTS:
%   prs = initial value of hyperparameters of ALDf prior cov. mat.
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
nsevar = prs(1);
prs = prs(2:end); % Hyper-params, except noise variance

% Check total dimension of the filter
leng_ndims = length(ndims);

%% form Cprior based on coordinate dimension

switch leng_ndims
    case (1)
        
        nx = ndims(1);
        
        M = prs(1); % unpack params
        vmu = prs(2);
        ovsc = prs(3);
        
        fx = FFTaxis(nx); % coordinates in Fourier domain
        w = fx(:);
        muvec = vmu.*ones(1, length(w));
        
    case (2)
        
        ny = ndims(1);
        nx = ndims(2);
        
        M = [prs(1) prs(2); prs(2) prs(3)]; % unpack params
        vmu = [prs(4); prs(5)];
        ovsc = prs(6);
        
        [fy, fx] =  FFT2axis(ny,nx);  % coordinates in Fourier domain
        w = [fy(:) fx(:)];
        muvec = repmat(vmu, 1, length(w));
        
    case (3)
        
        nt = ndims(1);
        ny = ndims(2);
        nx = ndims(3);
        
        M = [prs(1) prs(2) prs(3); prs(2) prs(4) prs(5); prs(3) prs(5) prs(6)]; % unpack params
        vmu = [prs(7); prs(8); prs(9)];
        ovsc = prs(10);
        
        
        [ft, fy, fx] =  FFT3axis(nt, ny, nx); % coordinates in Fourier domain
        w = [ft(:) fy(:) fx(:)];
        muvec = repmat(vmu, 1, length(w));
        
    otherwise
        
        disp('this dimension is not applicable');
        
end

absMw = abs(M*w');
sign_absMw = sign(M*w')';
X2 = (absMw - muvec)';
cdiag = exp(-.5*sum(X2.*X2,2)- ovsc); % diag of ALDf prior cov. mat.

svMin = 1e-6; % singular value threshold for eliminating dimensions  to make code fast
svthresh = max(cdiag)*svMin;

% prune data based on frequency sparsity
if min(cdiag)>svthresh
    iikeep = true(length(cdiag), 1);
else
    iikeep = (cdiag>=svthresh);
    X2 = X2(iikeep, :);
    w = w(iikeep, :);
    sign_absMw = sign_absMw(iikeep, :);
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
if leng_ndims==1
    nM = leng_ndims;
    der_absMw = zeros(leng_keep, leng_ndims, nM);
    der_absMw(:,:,1) = sign_absMw.*w;
else
    nM = leng_ndims + nchoosek(leng_ndims, 2);
    der_absMw = zeros(leng_keep, leng_ndims, nM);
    if leng_ndims == 2
        der_absMw(:,:,1) = sign_absMw.*[w(:,1) zeros(leng_keep,1)];
        der_absMw(:,:,2) = sign_absMw.*[w(:,2) w(:,1)];
        der_absMw(:,:,3) = sign_absMw.*[zeros(leng_keep,1) w(:,2)];
    else %leng_ndims ==3
        der_absMw(:,:,1) = sign_absMw.*[w(:,1) zeros(leng_keep,2)];
        der_absMw(:,:,2) = sign_absMw.*[w(:,2) w(:,1) zeros(leng_keep,1)];
        der_absMw(:,:,3) = sign_absMw.*[w(:,3) zeros(leng_keep,1) w(:,1)];
        der_absMw(:,:,4) = sign_absMw.*[zeros(leng_keep,1) w(:,2) zeros(leng_keep,1)];
        der_absMw(:,:,5) = sign_absMw.*[zeros(leng_keep,1) w(:,3) w(:,2)];
        der_absMw(:,:,6) = sign_absMw.*[zeros(leng_keep,2) w(:,3)];
    end
end

% gradients wrt M
for i=1:nM
    cGrad(:,i) = - sum(X2.*der_absMw(:,:,i),2);
end

% gradients wrt mu
der_mu = zeros(leng_keep, leng_ndims, leng_ndims);

for i=1:leng_ndims
    der_mu(:,i,i) = - I;
    cGrad(:,nM+i) =  - sum(X2.*der_mu(:, :, i), 2);
end

% gradients wrt overall scale
cGrad(:,end) = - I;

%% Gradients of evidence w.r.t. hyperparams

Ivec = ones(leng_keep, 1);
diagTrm = Ivec - diag(LLinvC) - sum((khat*(XY(iikeep))').*transpose(LLinvC), 2)./nsevar;
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


