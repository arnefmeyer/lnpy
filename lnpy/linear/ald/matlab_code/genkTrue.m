function kTrue = genkTrue(nDims)
%
%-------------------------------------------------------------------------
% genkTrue.m: generate kTrue for simulations with predetermined parameters
%-------------------------------------------------------------------------
% 
% INPUT ARGUMENTS:
%   nDims   a vector for filter dimensions
%               e.g.) 1-d, nDims = nX
%                     2-d, nDims = [nT, nX]
%                     3-d, nDims = [nT, nY, nX]
%
% OUTPUT ARGUMENTS:
%   kTrue  a true filter in a given dimension
% 
% Updated: 25/12/2011 Mijung Park

lengDims = length(nDims);

if lengDims == 1
    
    nX = nDims;
    sd1 = nX/15; sd2 = nX/10; % standard deviations for two Gaussians.
    xx = (1:nX)';
    kTrue = normpdf(xx,nX/2,sd1) - normpdf(xx,nX/2,sd2);
    kTrue = kTrue/norm(kTrue);
    
elseif lengDims == 2
    
    nT = nDims(1); 
    nX = nDims(2); % spatial and time dimension
    [xx,yy] = meshgrid(1:nX,1:nT); % x and t coordinates
    pxPerCycle = 6;     % pixels per cycle
    ori = pi/4;          % Orientation
    sf = 1./pxPerCycle;  % Spatial frequ ency
    phase = pi/2;        % Phase.  (0 = sine; pi/2 = cosine)
    mu = [nX/2, nT/2];    % Mean (center location)
    stdevs = [nX.*0.1, nT.*0.1];  % stdev along Gaussian major & minor axis, respectively
%     stdevs = [1.5, 1.5];
    kTrue = makeGabor(ori,sf,phase,mu,stdevs,xx,yy);
    
elseif lengDims == 3
    nT = nDims(1);
    nY = nDims(2); 
    nX = nDims(3); 
    
    % ktemp: temporal filter
    sdfilt = 1.5;
    ktemp = -finitediff(finitediff(exp(-((1:nT)-nT/3).^2/(2*sdfilt.^2))'));
    ktemp = ktemp./norm(ktemp);
    
    % kspatial: spatial filter
    x1 = 1:nX; y1 = 1:nY;
    [xx,yy] = meshgrid(x1,y1); % x and y coordinates
    pxPerCycle = 6;     % pixels per cycle
    ori = pi/4;          % Orientation
    sf = 1./pxPerCycle;  % Spatial frequ ency
    phase = pi/2;        % Phase.  (0 = sine; pi/2 = cosine)
    mu = [nX/2,nY/2];    % Mean (center location)
    stdevs = [nY*0.1,nX*0.2];  % stdev along Gaussian major & minor axis, respectively
    kspatial = makeGabor(ori,sf,phase,mu,stdevs,xx,yy);
    
    % combination of ktemp and kspatial
    kTrue = zeros(nY,nX,nT);
    for i=1:nT
        kTrue(:,:,i) = kspatial.* ktemp(i);
    end
    
else
    disp('Wrong dimension');
end