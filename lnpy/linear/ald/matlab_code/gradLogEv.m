function [logev, khat, LL, dLdv, ddLddv, LLinvC, diag_LLXXLLinvC] = gradLogEv(Cprior, nsevar,XX,XY,YY,ny)
%
%--------------------------------------------------------------------------
% gradLogEv.m: compute log-evidence and its derivatives wrt noise variance
%--------------------------------------------------------------------------
%
% INPUT ARGUMENTS: 
%   CpriorInv = inverse covariance matrix for prior
%   nsevar = variance of likelihood
%   X = design matrix
%   Y = dependent variable
%   XX = X'*X;  (alternative to passing X & Y)
%   YY = Y'*Y;  (alternative to passing X & Y)
%   ny = length(Y);  
%
% OUTPUT ARGUMENTS: 
%   logev = log-evidence
%   khat = MAP estimate (posterior mean)
%   LL = posterior covariance matrix
%   dLdv = deriv of log-ev with respect to residual variance
%   ddLddv = 2nd deriv of log-ev w.r.t to residual variance 
%   LLinvC = LL*inv(C), which we need to compute Hessian
%   diag_LLXXLLinvC = diag(LL*XX*LL*inv(C)), which we need to compute Hessian
%
%  (Updated: 23/12/2011 Mijung Park) 


nx = size(XX,1);
I = speye(nx);

% to make sure the diagonal of XX is real
if ~isreal(diag(XX))
    XX = tril(XX, -1) + diag(real(diag(XX))) + triu(XX, 1);
end

CXX = Cprior*XX/nsevar;

% 1. Compute log-evidence
% log-determinant term
if isdiag(Cprior)  % Special case if Cprior is diagonal 
   Csqrt = sqrt(Cprior);
    trm1 = -.5*(logdet(real(Csqrt*XX*Csqrt/nsevar + I)) + (ny)*log(2*pi*nsevar));
else % If non-diagonal
    trm1 = -.5*(logdetns(real(XX*Cprior/nsevar+I)) + (ny)*log(2*pi*nsevar));
end

LLinvC = (CXX+I)\I; % need this for computing derivatives w.r.t. noise variance

trm2 = -.5*(YY/nsevar - real(XY'*(LLinvC*(Cprior*XY)))/nsevar.^2); % 'real': because of rounding error...

logev = trm1+trm2;

% 2. Compute posterior variance
LL = LLinvC*Cprior; % Posterior Covariance

% 3.  Compute posterior mean
khat = LL*XY/nsevar;  % Posterior Mean (MAP estimate)

if nargout > 3  % Compute gradient info
 
   residErr = YY - 2*khat'*XY + khat'*XX*khat;
   gradtrm1 = nx - ny - trace(LLinvC);
     
   % 1st term
   df1 = 0.5*(1/nsevar)*gradtrm1;
   diag_LLXXLLinvC = sum((LL*XX).*transpose(LLinvC),2);
   traceOfLLXXLLinvC = sum(diag_LLXXLLinvC); % trace(LL*XX*LLinvC)
   ddf1 = -0.5*(1/nsevar.^2)*gradtrm1 - 0.5*(1/nsevar.^3)*traceOfLLXXLLinvC;

   % 2nd term
   df2 = 0.5*(1/nsevar^2)*residErr;
   dkdv = -(1/nsevar)*LLinvC*khat;  % Derivative of k_map (posterior mean) w.r.t nsevar
   ddf2 = -(1/nsevar^3)*residErr +(1/nsevar^2)*(- XY'*dkdv + khat'*XX*dkdv);

   dLdv = df1 + df2;
   ddLddv = ddf1 + ddf2;
   
end


    