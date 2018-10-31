function [f, df, khat, LL] = gradPrior_ALDsf(prs, datastruct)
%
%--------------------------------------------------------------------------
% gradPrior_ALDsf: compute gradients of ALDsf prior covariance matrix
%--------------------------------------------------------------------------
% 
% INPUT ARGUMENTS:
%   prs = initial value of hyperparameters of ALDsf prior cov. mat.
%   datastruct - has data info such as XX, XY, YY, stimulus dimension
%
% OUTPUT ARGUMENTS:
%   f - maximum evidence given data
%   df - df/dtheta at f = f_max
%   khat - estimate of kernel
%   LL - posterior covariance at f = f_max
%
%  (Updated: 25/12/2011 Mijung Park) 

% Unpack data
XX = datastruct.xx;
XY = datastruct.xy;
YY = datastruct.yy;
nsamps = datastruct.nstim;
ndims = datastruct.ndims;

% Unpack params
nsevar = abs(prs(1)); % to avoid nsevar<0
prs = prs(2:end); % Hyper-params, except noise variance

% Check total dimension of the filter
leng_ndims = length(ndims);

%% form Cprior based on coordinate dimension

switch leng_ndims
    case (1)
        
        nx = ndims(1);
        
        % ALDs
        numbTheta_s = length(prs(1:2)); % # of parameters in ALDs prior cov. mat.
        
        v_mu = prs(1); % unpack params
        v_gam = prs(2);
        
        L = v_gam.^2; % precision
        X = [1:nx]'; % coordinate
        X2 = X - repmat(v_mu', size(X,1),1); % move toward the mean
        
        X2Li = X2/L; % X2*inv(L)
        LiX2 = L\X2'; % inv(L)*X2
        
        vX = exp(-.5*sum((X2Li).*X2,2)); % diagonal of ALDs prior cov. mat.
        
        % ALDf
        numbTheta_f = length(prs(3:4)); % # of parameters in ALDf prior cov. mat.
        
        M = prs(3); % unpack params
        vmu = prs(4);
        
        fx = FFTaxis(nx); % coordinate in Fourier domain
        
        w = fx(:);
        muvec = vmu.*ones(length(w),1);
        
        absMw = abs(M*w);
        W2 = absMw - muvec; % move toward the mean
        sign_absMw = sign(M*w')'; % for computing derivatives
        
        vF = exp(-.5*W2.^2); % diagonal of ALDf prior cov. mat.
        
        nrmtrm = sqrt(nx);  % normalization term to make FFT orthogonal
        BB = FFTmatrix(nx)/nrmtrm;
        
    case (2)
        
        nt = ndims(1);
        nx = ndims(2);
        
        % ALDs
        numbTheta_s = length(prs(1:5)); % # of parameters in ALDs prior cov. mat.
        
        v_mu = prs(1:2); % unpack params
        v_gam = prs(3:4);
        v_phi = prs(5);
        
        
        [t1, x1] = ndgrid(1:nt,1:nx); % coordinates
        X = [t1(:), x1(:)];
        X2 = X - repmat(v_mu', size(X,1),1); % move toward the mean
        
        inds = find(1-tril(ones(leng_ndims))); % make precision matrix L
        Lmult = ones(leng_ndims);
        Lmult(inds) = v_phi;
        Lmult = Lmult.*Lmult';
        L = v_gam*v_gam';
        L = L.*Lmult;
        
        X2Li = X2/L;
        LiX2 = L\X2';
        
        vX = exp(-.5*sum((X2Li).*X2,2)); % diagonal of ALDs prior cov. matrix.
        
        % ALDf
        numbTheta_f = length(prs(6:10)); % # of parameters in ALDf prior cov. mat.
        
        M = [prs(6) prs(7); prs(7) prs(8)]; % unpack params
        vmu = [prs(9); prs(10)];
        
        [ft, fx] =  FFT2axis(nt,nx); % coordinates in Fourier domain
        w = [ft(:) fx(:)];
        
        muvec = repmat(vmu, 1, length(w));
        absMw = abs(M*w');
        W2 = (absMw - muvec)'; % move toward the mean
        vF = exp(-.5*sum(W2.*W2,2)); % diagonal of ALDf prior cov. matrix.
        
        sign_absMw = sign(M*w')'; % for computing derivatives
        
        nrmtrm = sqrt(nt*nx);  % normalization term to make FFT orthogonal
        BB = FFT2matrix(nt,nx)/nrmtrm;
        
    case (3)
        
        nt = ndims(1);
        ny = ndims(2);
        nx = ndims(3);
        
        % ALDs
        numbTheta_s = length(prs(1:9)); % # of parameters in ALDs prior cov. mat.
        
        v_mu = prs(1:3);
        v_gam = prs(4:6);
        v_phi = prs(7:9);
        
        [t1, y1, x1] = ndgrid(1:nt,1:ny,1:nx); % coordinates
        X = [t1(:), y1(:), x1(:)];
        
        X2 = X - repmat(v_mu', size(X,1),1); % move toward the mean
        
        inds = find(1-tril(ones(leng_ndims))); % make precision matrix L
        Lmult = ones(leng_ndims);
        Lmult(inds) = v_phi;
        Lmult = Lmult.*Lmult';
        L = v_gam*v_gam';
        L = L.*Lmult;
        
        X2Li = X2/L;
        LiX2 = L\X2';
        
        vX = exp(-.5*sum((X2Li).*X2,2)); % diagonal of ALDs prior cov. matrix.

        % ALDf
        numbTheta_f = length(prs(10:18)); % # of parameters in ALDf prior cov. mat.

        M = [prs(10) prs(11) prs(12); prs(11) prs(13) prs(14); prs(12) prs(14) prs(15)]; % unpack params
        vmu = [prs(16); prs(17); prs(18)];
                
        [ft, fy, fx] =  FFT3axis(nt, ny, nx); % coordinates in Fourier domain
        w = [ft(:) fy(:) fx(:)];
        
        muvec = repmat(vmu, 1, length(w));
        absMw = abs(M*w');
        W2 = (absMw - muvec)'; % move toward the mean
        vF = exp(-.5*sum(W2.*W2,2)); % diagonal of ALDf prior cov. matrix.
        
        sign_absMw = sign(M*w')'; % for computing derivatives
        
        nrmtrm = sqrt(nt*ny*nx);  % normalization term to make FFT orthogonal
        BB = FFT3matrix(nt, ny, nx)/nrmtrm;

        
    otherwise
        
        disp('this dimension is not applicable');
        
end

ovsc_sf = prs(end);

svMin = 1e-6; % singular value threshold for eliminating dimensions  to make code fast
svMax = 1e12;
vX(isinf(vX)) = svMax;
vF(isinf(vF)) = svMax; % if max(vX) = Inf, then svthreshX = inf
svthreshX = max(vX)*svMin;
svthreshF = max(vF)*svMin;

iix = (vX>=svthreshX);
iif = (vF>=svthreshF);
nix = sum(iix);
nif = sum(iif);

% prune data based on pixel and frequency sparsity
vX = vX(iix);
vF = vF(iif);
bb = BB(iif, iix);
XX = XX(iix, iix);
XY = XY(iix);
X2Li = X2Li(iix, :);
LiX2 = LiX2(:, iix);
W2 = W2(iif, :);
w = w(iif, :);
sign_absMw = sign_absMw(iif, :);

firstTrm = exp(-ovsc_sf)*diag(sqrt(vX))*bb';
secondTrm = diag(vF)*bb*diag(sqrt(vX));
Cprior = firstTrm*secondTrm;
% Cprior = exp(-ovsc_sf)*diag(sqrt(vX))*BB*diag(vF)*BB'*diag(sqrt(vX));

%% evaluate log-evidence & compute 1st/2nd Derivatives w.r.t. noise variance

[f, khat, LL, df1, ddf1, LLinvC] = gradLogEv(Cprior, abs(nsevar),XX,XY,YY,nsamps);
f = -f;

numb_params = length(prs); % total number of parameters in ALDsf prior cov. mat.
I = ones(nif,1);

dCs_dtheta = zeros(nix, numbTheta_s); % dCs/dtheta_s
dCf_dtheta = zeros(nif, numbTheta_f); % dCf/dtheta_f

%% Gradients of Cprior w.r.t each hyperparam

% gradients wrt v_mu in vX
der_vmu = zeros(nix, leng_ndims, leng_ndims);
der_mu = - ones(nix,1);

% gradients wrt  v_gam in vX
lg = length(v_gam); % # of gammas
der_gam = zeros(lg, lg, lg); % dL/dgamma
% prepare der. wrt each gam
for i=1:lg
    Zeromat = zeros(lg, lg);
    Zeromat(i,:) = L(i,:)./v_gam(i);
    Zeromat(:,i) = L(:,i)./v_gam(i);
    Zeromat(i,i) = 2*v_gam(i);
    der_gam(:,:,i) = Zeromat; % dL/dgamma
end

% gradients wrt M in vF
if leng_ndims==1
    nM = leng_ndims;
    der_absMw = zeros(nif, leng_ndims, nM);
    der_absMw(:,:,1) = sign_absMw.*w; % dM/dm_1
else
    nM = leng_ndims + nchoosek(leng_ndims, 2);
    der_absMw = zeros(nif, leng_ndims, nM);
    if leng_ndims == 2
        der_absMw(:,:,1) = sign_absMw.*[w(:,1) zeros(nif,1)]; % dM/dm_1
        der_absMw(:,:,2) = sign_absMw.*[w(:,2) w(:,1)]; % dM/dm_2
        der_absMw(:,:,3) = sign_absMw.*[zeros(nif,1) w(:,2)]; % dM/dm_3
    else %leng_ndims ==3
        der_absMw(:,:,1) = sign_absMw.*[w(:,1) zeros(nif,2)]; % dM/dm_1
        der_absMw(:,:,2) = sign_absMw.*[w(:,2) w(:,1) zeros(nif,1)]; % dM/dm_2
        der_absMw(:,:,3) = sign_absMw.*[w(:,3) zeros(nif,1) w(:,1)]; % dM/dm_3
        der_absMw(:,:,4) = sign_absMw.*[zeros(nif,1) w(:,2) zeros(nif,1)]; % dM/dm_4
        der_absMw(:,:,5) = sign_absMw.*[zeros(nif,1) w(:,3) w(:,2)]; % dM/dm_5
        der_absMw(:,:,6) = sign_absMw.*[zeros(nif,2) w(:,3)]; % dM/dm_6
    end
end

% store dCf/dM to dCf/dtheta_f
for i=1:nM
    dCf_dtheta(:,i) = - sum(W2.*der_absMw(:,:,i),2);
end

% gradients wrt mu_f in vF
der_muf = zeros(nif, leng_ndims, leng_ndims);
j=1;
for i=1:leng_ndims
    der_vmu(:,i,j) = der_mu;
    j=j+1;
    dCs_dtheta(:,i) = - 0.5*diag(X2Li*der_vmu(:,:,i)'); % store dCs/dmu to dCs/dtheta_s
    dCs_dtheta(:,leng_ndims+i) =  0.25*diag(X2Li*der_gam(:,:,i)*LiX2); % store dCs/dgamma to dCs/dtheta_s
    der_muf(:,i,i) = - I;
    dCf_dtheta(:,nM+i) =  - sum(W2.*der_muf(:, :, i), 2); % store dCf/dmu_f to dCf/dtheta_f
end

% gradients wrt  v_phi in vX
if leng_ndims ~=1
    lp = length(v_phi);
    der_phi = zeros(leng_ndims, leng_ndims, lp);
    gamL = v_gam*v_gam';
    inds_gamL = gamL(inds);
    for i=1:lp
        B = zeros(leng_ndims, leng_ndims);
        B(inds(i)) = inds_gamL(i);
        der_phi(:,:,i) = B+B';
    end
    for i=1:lp
        dCs_dtheta(:,2*leng_ndims+i) = 0.25*diag(X2Li*der_phi(:,:,i)*LiX2); % store dCs/dphi to dCs/dtheta_s
    end
end

%% Gradients of evidence w.r.t. hyperparams

LLinvCtr = transpose(LLinvC);
dfdthet = zeros(length(prs),1);
trm1 = 0.5/(nsevar^2)*XY'*LLinvC;
trm2 = LLinvCtr*XY;

for i=1:numb_params
    if i<numbTheta_s+1
        cprime =  2*bsxfun(@times, Cprior, dCs_dtheta(:,i)');
    elseif (numbTheta_s<i)&&(i<length(prs))
        cprime = bsxfun(@times,  firstTrm, dCf_dtheta(:,i-numbTheta_s)')*secondTrm;
    else % i==length(prs)
        cprime =  - Cprior;
    end
    dfdthetTrm1 = -0.5*1/nsevar*sum(sum((LLinvC*cprime).*transpose(XX), 2));
    dfdthetTrm2 = trm1*cprime*trm2;
    
    dfdthet(i) = dfdthetTrm1 + dfdthetTrm2;
end
df = -[df1; dfdthet];
df = real(df);

%% project back to the original space

nx = prod(ndims);

khatreduced = khat; % khat (posterior mean)
khat = zeros(nx, 1);
khat(iix) = khatreduced;

Lmat = zeros(nx, nx); % LL (posterior covariance)
Lmat(iix, iix) = LL;
LL = Lmat;





