function [XX,XY] = fastCrossCov(X, Y, nkt)
%  fastCrossCov - computes auto- and cross-covariance X'X and X'Y for auto-regressive model
%
%  [XX,XY] = fastCrossCov(X, Y, nkt)
%
%  Computes the terms (X'*X) and (X'*Y) necessary to causally regress
%  dependent variable Y against regressors X
%  (i.e. Y(j) regressed against X(j-[0:nkt-1], :) 
%
%  Input:  
%      X [nT x nX] - matrix of design variables (vertical dimension is time)
%      Y [nT x  1] - output variable 
%    nkt [ 1 x  1] - number of time samples of X to use to predict Y.
%
%  Output:
%      XY [nX*nkt x 1]      - projection of Y onto X'.
%      XX [nX*nkt x nX*nkt] - stimulus auto-covariance X'*X;
%
%  Notes:
%  - Standard regression solution for weights is then K_ML = XX\XY; 
%  - Pads X with zeros for earliest values of Y
%  - Error checking: should give same output as
%                    M=makeStimRows(X,nkt); XX=M'*M; XY = M'*Y;
% 
%  (Updated: 14/03/2011 JW Pillow) 

[nT,nX] = size(X);
xlen = nX*nkt;

% Allocate space
XX = zeros(xlen,xlen);
XY = zeros(nkt,nX);

% Compute Terms
for jshft = 1:nkt
    
    % Cross-Covariance Term
    XY(nkt-jshft+1,:) = Y(jshft:nT)'*X(1:nT-jshft+1,:);
    
    % Auto-Covariance term 
    aa = X(1:nT-jshft+1,:)'*X(jshft:nT,:);
    ii = (xlen-(nX*jshft)+1):(xlen-nX*(jshft-1));  % some indices
    for jtrm = 1:nkt-jshft+1
        iirow = ii-(jtrm-1)*nX;
        iicol = ii-(jtrm-jshft)*nX;
        bb = X((nT-jtrm+2:nT)-jshft+1,:)'*X(nT-jtrm+2:nT,:);  % subtract off for zero-shifted rows
        XX(iirow,iicol) = aa-bb;
        if jshft>1 % replicate off-diagonal blocks
            XX(iicol,iirow) = XX(iirow,iicol)';
        end
    end
end

% Reshape XY term
XY = XY(:);

% % Reshape XX term
ii = reshape(reshape(1:xlen,nX,nkt)',xlen,1);
XX = XX(ii,ii);
