function [cGrad, iikeep, cHess, cdiag] = gradPrior_ALDs(prs, XX, ndims)
%
%--------------------------------------------------------------------------
% gradPrior_ALDs: compute gradients of ALDs prior covariance matrix
%--------------------------------------------------------------------------
% 
% INPUT ARGUMENTS:
%   prs - given hyperparameters for ALDs prior cov. mat.
%   XX - x'*x
%   ndims - input dimensions
%
% OUTPUT ARGUMENTS:
%   cGrad - gradients of ALDs prior covariance matrix wrt hyperparameters
%   iikeep - index of points to keep after thresholding
%   cHess - Hessian of ALDs prior covariance matrix wrt hyperparameters
%   cdiag - diagonal of ALDs prior covariance matrix
%
%  (Updated: 23/12/2011 Mijung Park) 


% Check total dimension of the filter
total_dim = size(XX,1);
leng_ndims = length(ndims);

switch leng_ndims
    case (1)
        nx = ndims;
        v_mu = prs(1);
        v_gam = prs(2);
        v_phi = 0;
        X = [1:nx]';
    case (2)
        nt = ndims(1);
        nx = ndims(2);
        v_mu = prs(1:2);
        v_gam = prs(3:4);
        v_phi = prs(5);
        [t1, x1] = ndgrid(1:nt,1:nx);
        X = [t1(:), x1(:)];
    case (3)
        nt = ndims(1);
        ny = ndims(2);
        nx = ndims(3);
        v_mu = prs(1:3);
        v_gam = prs(4:6);
        v_phi = prs(7:9);
        [t1, y1, x1] = ndgrid(1:nt,1:ny,1:nx);
        X = [t1(:), y1(:), x1(:)];
    otherwise
        disp('this dimension is not applicable');
end

X2 = X - repmat(v_mu', size(X,1),1); % move toward the mean

if leng_ndims ==1
    L = v_gam.^2;
else
    % make precision matrix L
    inds = find(1-tril(ones(leng_ndims)));
    Lmult = ones(leng_ndims);
    Lmult(inds) = v_phi;
    Lmult = Lmult.*Lmult';
    L = v_gam*v_gam';
    L = L.*Lmult;
end

numb_params = length(prs); % alpha, mu, gam, phi
leng_X = length(X);

alpha = prs(end); % overall scaler
cdiag = reshape(exp(-alpha -.5*sum((X2/L).*X2,2)), [], 1); % diagonal of prior cov matrix

%% Check for whether prior is full rank (based on condition number)

svMin = 1e-6; % singular value threshold for eliminating dimensions
svthresh = max(cdiag)*svMin;

if min(cdiag)>svthresh
    iikeep = true(total_dim,1);
else
    iikeep = (cdiag>=svthresh); % pixels to keep
    if leng_ndims ==1
        X2 = X2(iikeep);
    else
        X2 = X2(iikeep,:);
    end
    cdiag = cdiag(iikeep);
    cdiag(isinf(cdiag)) = 1e20;
    % because 1/Inf=0 -> cInv becomes all zero -> cannot compute
    % logdet(cInv)
end

leng_keep = sum(iikeep);

X2Li = X2/L;
LiX2 = L\X2';

%% Gradients of C w.r.t. theta

cGrad = zeros(leng_keep, numb_params); % here cGrad is dC/dtheta*invC
I = ones(leng_keep,1);

% gradients of vector mu (vmu) wrt mu_i
der_vmu = zeros(leng_keep, leng_ndims, leng_ndims);
der_mu = - ones(leng_keep,1); 
j=1;
for i=1:leng_ndims
    der_vmu(:,i,j) = der_mu; % dvmu/dmu_i
    j=j+1;
end

% gradients of L wrt v_gam
lg = length(v_gam);
der_gam = zeros(lg, lg, lg);
for i=1:lg
    Zeromat = zeros(lg, lg);
    Zeromat(i,:) = L(i,:)./v_gam(i);
    Zeromat(:,i) = L(:,i)./v_gam(i);
    Zeromat(i,i) = 2*v_gam(i);
    der_gam(:,:,i) = Zeromat; % dL/dgam
end

% grad. wrt vector gam and vector mu
for i=1:leng_ndims
    cGrad(:,i) = - diag(X2Li*der_vmu(:,:,i)'); % dC/dmu
    cGrad(:,leng_ndims+i) =  diag(.5.*X2Li*der_gam(:,:,i)*LiX2); % dC/dgam
end

% gradients of L wrt  v_phi
if leng_ndims ~=1 % 1d case doesn't have cross-terms in L
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
        cGrad(:,2*leng_ndims+i) = diag(.5.*X2Li*der_phi(:,:,i)*LiX2); % dC/dv_phi
    end
    
end

% gradient wrt overall scale (alpha)
cGrad(:,end) = - I;

%% Hessian: organized as lower triagonal

numb_H_terms = numb_params + nchoosek(numb_params, 2);
cHess = zeros(leng_keep, numb_H_terms);
lm = length(v_mu);
la = length(alpha);

%  1 dim.
if leng_ndims == 1
    
    cHess(:,1) = - diag(der_vmu*(L\der_vmu')) - diag(X2Li*der_vmu').*cGrad(:,1); % d^2C/dmu^2
    
    cHess(:,2) = diag(X2Li*der_gam*(L\der_vmu')) - diag(X2Li*der_vmu').*cGrad(:,2); % d^2C/dmudgam
    
    cHess(:,3) =  - diag(X2Li*der_vmu').*cGrad(:,end); % d^2C/dmudalpha
    
    cHess(:,4) =  - diag(.5*X2Li*( der_gam*(L\der_gam) - 2 + der_gam*(L\der_gam))*LiX2 ) + ...
        diag(.5*X2Li*der_gam*LiX2).*cGrad(:,2); % d^2C/dgam^2
    
    cHess(:,5) = diag(.5*X2Li*der_gam*LiX2).*cGrad(:,end); % d^2C/dgamdalpha
    
    cHess(:,6) = I; % d^2C/dalphadalpha
    
else % leng_ndims == 2 or 3

    % d^2C/dmu^2 and d^2C/dgam^2
    H_MuMu = zeros(lm, lm, leng_keep); 
    H_GamGam = zeros(lg, lg, leng_keep); 
    
    for j=1:leng_ndims
        for i=j:leng_ndims
            if i==j
                dgam_dgam = zeros(leng_ndims, leng_ndims);
                dgam_dgam(i,j) =2;
            else
                if leng_ndims==2
                    indss = find(der_phi(:,:,1));
                    dgam_dgam = zeros(leng_ndims, leng_ndims);
                    dgam_dgam(indss) = v_phi(1);
                else
                    indss = find(der_phi(:,:,mod(i+j,leng_ndims)+1));
                    dgam_dgam = zeros(leng_ndims, leng_ndims);
                    dgam_dgam(indss) = v_phi(mod(i+j,leng_ndims)+1);
                end
            end
            H_MuMu( j , i , :) = - diag(der_vmu(:,:,i)*(L\der_vmu(:,:,j)')) - diag(X2Li*der_vmu(:,:,j)').*cGrad(:,i); 
            H_GamGam( j,  i, :) = - diag(.5*X2Li*(der_gam(:,:,i)*(L\der_gam(:,:,j)) - dgam_dgam + der_gam(:,:,j)*(L\der_gam(:,:,i)))*LiX2) + ...
                diag(.5*X2Li*der_gam(:,:,j)*LiX2).*cGrad(:,leng_ndims+i);
        end
    end
    
    % d^2C/dmu_dgam
    H_MuGam = zeros(lm, lm, leng_keep); 
    for j=1:leng_ndims
        for i=1:leng_ndims
            H_MuGam( j, i, :) = diag(X2Li*der_gam(:,:,i)*(L\der_vmu(:,:,j)')) - diag(X2Li*der_vmu(:,:,j)').*cGrad(:,leng_ndims+i);
        end
    end
    
    % d^2C/dmu_dphi
    H_MuPhi = zeros(lm, lp, leng_keep);  
    for j=1:leng_ndims
        if leng_ndims==2
            H_MuPhi( j, 1, :) = diag(X2Li*der_phi(:,:,1)*(L\der_vmu(:,:,j)')) - diag(X2Li*der_vmu(:,:,j)').*cGrad(:,2*leng_ndims+1);
        else
            for i=1:leng_ndims
                H_MuPhi( j, i, :) = diag(X2Li*der_phi(:,:,i)*(L\der_vmu(:,:,j)')) - diag(X2Li*der_vmu(:,:,j)').*cGrad(:,2*leng_ndims+i);
            end
        end
    end
    
    % d^2C/dmu_dalpha and d^2C/dgam_dalpha
    H_MuAlpha = zeros(lm, la, leng_keep);
    H_GamAlpha = zeros(lm, la, leng_keep); 
    for j=1:leng_ndims
        H_MuAlpha(j, la, :) = - diag(X2Li*der_vmu(:,:,j)').*cGrad(:,end);
        H_GamAlpha(j, la, :) = diag(.5*X2Li*der_gam(:,:,j)*LiX2).*cGrad(:,end);
    end
    
    % d^2C/dgam_dphi
    H_GamPhi = zeros(lg, lp, leng_keep);
    for j=1:lg
        for i=1:lp
            if i+j==4
                dgam_dphi = zeros(leng_ndims, leng_ndims);
            else
                dgam_dphi = der_phi(:,:,i)./v_gam(j);
            end
            H_GamPhi(j,i,:) = - diag(.5*X2Li*( der_phi(:,:,i)*(L\der_gam(:,:,j)) - dgam_dphi + der_gam(:,:,j)*(L\der_phi(:,:,i)))*LiX2) + ...
                diag(.5*X2Li*der_gam(:,:,j)*LiX2).*cGrad(:,2*leng_ndims+i);
        end
    end
    
    % d^2C/dphi^2
    H_PhiPhi = zeros(lp, lp, leng_keep);
    for j=1:lp
        for i=j:lp
            if leng_ndims ==2
                H_PhiPhi(j, i, :) = - 2*diag(.5*X2Li*der_phi(:,:,j)*(L\der_phi(:,:,i))*LiX2)+ ...
                    diag(.5*X2Li*der_phi(:,:,j)*LiX2).*cGrad(:,2*leng_ndims+i);
            else % leng_ndims ==3
                H_PhiPhi(j, i, :) = - diag(.5*X2Li*(der_phi(:,:,j)*(L\der_phi(:,:,i)) + der_phi(:,:,i)*(L\der_phi(:,:,j)))*LiX2) + ...
                    diag(.5*X2Li*der_phi(:,:,j)*LiX2).*cGrad(:,2*leng_ndims+i);
            end
        end
    end
    
    % d^2C/dphi_dalpha
    H_PhiAlpha = zeros(lp, la, leng_keep);
    for j=1:lp
        H_PhiAlpha(j, 1, :) = diag(.5*X2Li*der_phi(:,:,j)*LiX2).*cGrad(:,end);
    end

    % d^2C/dalpha^2
    H_AlphAlph = zeros(la, la, leng_keep);
    H_AlphAlph(la, la, :) = I;

    % put everything in Hmat (size: #param by # param by length(X))   
    Hmat = [H_MuMu H_MuGam H_MuPhi H_MuAlpha ; zeros(lg, lg, leng_keep) H_GamGam H_GamPhi H_GamAlpha; zeros(lp, lm+lg, leng_keep) H_PhiPhi H_PhiAlpha ; zeros(la, numb_params-1, leng_keep) H_AlphAlph];
    
    % reshape Hmat to cHess (size: length(X) by upper triangular part of Hmat)
    Hmat_tr = permute(Hmat, [2 1 3]);
    trilaaa = find(tril(ones(numb_params, numb_params), 0)); % lower triangular matrix including diagonal
    Hmat_tr_rsh = reshape(Hmat_tr, [], 1, leng_keep);
    Hmat_tr_rsh2 = permute(Hmat_tr_rsh(trilaaa,1,:), [3 1 2]);
    cHess(:, 1:numb_H_terms) = Hmat_tr_rsh2(:,1:numb_H_terms);
    
end

