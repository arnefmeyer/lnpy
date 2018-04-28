function [mu,L] = compPostMeanVar(X,Y,nsevar,CpriorInv,XX,XY)
% [mu,L] = compPostMeanVar(X,Y,nsevar,CpriorInv)
%     or
% [mu,L] = compPostMeanVar([],[],nsevar,CpriorInv,XX,XY)
%
% Compute posterior mean and variance given observed data Y given X, 
% (Or, given xx = X'*X; XY = X'*Y) and with
% noise variance nsevar and prior inverse-covariance CpriorInv


if nargin <= 4
    XX = X'*X;
    XY = X'*Y;
end

L = inv(XX./nsevar + CpriorInv);  % covariance

mu = L*(XY)/nsevar;  % mean