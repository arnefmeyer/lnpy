function [datastruct] = formDataStruct(x, y, nkt, spatialdims)

%--------------------------------------------------------------------------
% formDataStruct.m: form data structure of sufficient statistics from raw data
%--------------------------------------------------------------------------
%
%  DESCRIPTION:
%    Compute sufficient statistics (x'*x, x'*y, y'*y both in spacetime &
%    Fourier domain) and store them in a data structure.
%
%  INPUT ARGUMENTS:
%   x = (nstim x nspatial) matrix of design variables (vertical dimension indicates time)
%   y = (nstim x 1) output variable (column vector)
%   nkt  = number of time samples of X to use to predict Y.
%   spatialdims = spatial dimension of input stimulus
%        e.g.) 1D: nX (nX>1 when nkt=1, or nX=1 when nkt>1)
%                2D: nX when nkt>1, or [nY; nX] when nkt=1
%                3D: [nY; nX] when nkt>1
%
%  OUPUT ARGUMENTS:
%  datastruct
%         datastruct.xx = x'*x
%         datastruct.xy = x'*y
%         datastruct.yy = y'*y
%         datastruct.xxf = x_f'*x_f, where x_f is x in Fourier domain
%         datastruct.xyf = x_f'*y
%         datastruct.nstim = length of stimulus
%         datastruct.nkt = # of time samps of x to predict y
%         datastruct.spatialdims = spatial dimension of stimulus
%         datastruct.ndims = actual dimension of stimulus
%
%  (Updated: 25/12/2011 Mijung Park)

% form data structure
[datastruct.xx, datastruct.xy] = fastCrossCov(x, y, nkt);
datastruct.yy = y'*y;
datastruct.nstim = length(y);
datastruct.nkt = nkt;
datastruct.spatialdims = spatialdims;

lengspatialdims = length(spatialdims);

if (lengspatialdims==1)
    
    if ((spatialdims==1)&&(nkt>1))||((spatialdims>1)&&(nkt==1))
        % 1D fft (space or time only)
        M = FFTmatrix(spatialdims);
        datastruct.xxf = M*datastruct.xx*M';
        datastruct.xyf = M*datastruct.xy;
        datastruct.ndims = max(nkt, spatialdims); % 1D stimulus
    elseif ((spatialdims>1)&&(nkt>1))
        % 2D fft (time by space)
        M = FFT2matrix(nkt,spatialdims);
        datastruct.xxf = M*datastruct.xx*M';
        datastruct.xyf = M*datastruct.xy;
        datastruct.ndims = [nkt; spatialdims]; % 2D stimulus
    end
    
elseif (lengspatialdims==2)
    
    if nkt==1
        % 2D fft (space by space)
        M = FFT2matrix(spatialdims(1),spatialdims(2));
        datastruct.xxf = M*datastruct.xx*M';
        datastruct.xyf = M*datastruct.xy;
        datastruct.ndims = spatialdims; % 2D stimulus
    elseif nkt>1
        % 3D fft (time by space by space)
        M = FFT3matrix(nkt, spatialdims(1), spatialdims(2)); % nkt by ny by nx
        datastruct.xxf = M*datastruct.xx*M';
        datastruct.xyf = M*datastruct.xy;
        datastruct.ndims = [nkt; spatialdims]; % 3D stimulus
    end
    
else
    fprintf('this package does not support stimuli in higher than 3D.');
    
end



