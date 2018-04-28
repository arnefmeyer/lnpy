function [f,df,ddf, khat, LL] = gradLogEv_ALDs(prs, datastruct)
%
%--------------------------------------------------------------------------
% gradLogEv_ALDs.m: compute gradients of neg-log-evidence with ALDs prior
%--------------------------------------------------------------------------
% 
% INPUT ARGUMENTS:
%   prs = given hyperparameters for ALDs prior cov. mat.
%   datastruct - has data info such as XX, XY, YY, stimulus dimension
%
% OUTPUT ARGUMENTS:
%   f - negative log-evidence given prs
%   df - first derivatives of neg-log-evid wrt hyperparameters
%   ddf - second derivatives of neg-log-evid wrt hyperparameters
%   khat - ALDs RF estimate given prs
%   LL - posterior covariance given prs
%
%  (Updated: 23/12/2011 Mijung Park) 

%  Unpack the data information:
XX = datastruct.xx;
XY = datastruct.xy;
YY = datastruct.yy; 
nsamps = datastruct.nstim;
ndims = datastruct.ndims;

% Unpack params
nsevar = abs(prs(1));
prs = prs(2:end); % Hyper-params, except noise variance

% Generate prior covariance matrix
[cGrad, iikeep, cHess, cdiag] = gradPrior_ALDs(prs, XX, ndims); % C is not included in the output
niikeep = sum(iikeep);

if (isempty(cGrad))||(nsevar <=0)    
    
    f=1e20;
    df=[];
    ddf=[];
    khat =[];
    LL = [];
    
else
    
    % evaluate log-evidence & compute 1st/2nd Derivatives w.r.t. noise variance
    Cprior = diag(cdiag);
    [f, khat, LL, df1, ddf1, LLinvC, diag_LLXXLLinvC] = gradLogEv(Cprior, nsevar,XX(iikeep, iikeep),XY(iikeep),YY,nsamps);
    f = -f;
    
    % ----- Compute Gradient --------
    
    XYkeep = XY(iikeep);
    nx = size(XX,2);
    Ivec = ones(niikeep, 1);
    diagTrm = Ivec - diag(LLinvC) - sum((khat*XYkeep').*transpose(LLinvC), 2)./nsevar; 
    dfdthet = - .5*cGrad'*diagTrm;  % 0.5* Tr(C - Lambda - k*k')*dCinv/dtheta);
    df = -[df1; dfdthet];
    
    % ----- Compute Hessian --------
        
        % Assemble the d/(dthet_i dthetj) portion
        nthet = length(prs);
        dLdiag = zeros(niikeep,nthet);
        dk = zeros(niikeep,nthet);
        ddfdthet1 = zeros(nthet,nthet);
        ddfdthet2 = zeros(nthet,nthet);
        for j = 1:nthet
            dLdiag(:,j) = sum(bsxfun(@times, LLinvC, cGrad(:,j)).*LLinvC', 2);
            dk(:,j) = XYkeep'*LLinvC*diag(cGrad(:,j))*LLinvC/nsevar;
        end
        
        cGradProducts = zeros(niikeep, size(cHess,2));
        ii =1;
        for i=1:nthet
            for j=i:nthet
                cGradProducts(:,ii) = cGrad(:,i).*cGrad(:,j);
                ii = ii + 1;
            end
        end
        
        Htrm = sum(repmat(diagTrm,1,size(cHess,2)).*(2*cGradProducts - cHess),1);
        itrm = 1;
        for j = 1:nthet
            ddfdthet1(:,j) = - sum((cGrad - dLdiag - 2*repmat(khat,1,nthet).*dk).*repmat(cGrad(:,j),1,nthet),1);
            ddfdthet2(j:nthet,j) = Htrm(itrm:(itrm+nthet-j));
            itrm = itrm+nthet-j+1;
        end
        
        ddfdthet2 = ddfdthet2 + tril(ddfdthet2,-1)';
        ddfdthet = 0.5*(ddfdthet1+ddfdthet2);
        
        % Assemble the d/(dthet dnoisevar)
        dLnse = (1./nsevar.^2)*diag_LLXXLLinvC; % deriv of LL wrt nsevar
        dknse = (-1./nsevar.^2)*XYkeep'*LLinvC*LLinvC;
        ddf_dthet_dnse = .5*sum(repmat(dLnse + 2*diag(khat*dknse),1,nthet).*cGrad,1)';

        ddf = -[[ddf1;ddf_dthet_dnse], [ddf_dthet_dnse'; ddfdthet]];

end 


% --- Map back to full space, if prior cov has reduced-rank -----
if (nargout >= 2) && (niikeep < nx)

    % khat (posterior mean)
    khatreduced = khat;
    khat = zeros(nx,1);
    khat(iikeep) = khatreduced;

    % LL (posterior covariance)
    Lmat = zeros(nx, nx);
    Lmat(iikeep,iikeep) = LL;
    LL = Lmat;

end
