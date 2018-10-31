function [Stim,wr] = make1Fstim_2D(yxdims,fcutoff,pow,nreps)
% [Stim,freqs] = make1Fstim_2D(yxdims,samprate,fcutoff,pow,nreps);
%
% Create a 1D Gaussian stimulus of length nx, with power spectrum that
% falls off as 1./f.^pow;
%
% Inputs: 
%   yxdims = [ny,nx], number of vertical and horiz elts in stimulus
%   fcutoff = low-frequency cutoff (stimulus will be white below this)
%             (Assumed sampling rate here is 1Hz spatial freq)  
%   pow = assumed exponent for fall-off of frequencies
%   nreps = # of repetitions.
%
% Outputs:
%   Stim = 1/F stimulus.  (Each column is a single 1/F stimulus)
%   freqs = vector of frequencies used when computing 1/|F|.^pow
%   
% Example call:
% % create 20000-sample stimulus with 1/F spectrum above .1 Hz
% > Stim = make1Fstim_1D(20000,1000,.1,1); 
%
% NOTE: if ny ~= nx, then fcutoff should be set to at least 1/min(nx,ny).
%       Otherwise, stimulus will not be isotropic (i.e., will have more
%       power in horizontal than veritcal frequencies, or vice versa)

% Process input args
if length(yxdims)==1
    nx = yxdims;
    ny = yxdims;
else
    ny = yxdims(1);
    nx = yxdims(2);
end

% Set up bins for spatial frequency
[x,y] = meshgrid(1:nx, 1:ny);
wx = mod(x+nx/2-1,nx)-nx/2;  % spatial freqs, y
wy = mod(y+ny/2-1,ny)-ny/2;  % spatial freqs, y
wx = wx./nx;
wy = wy./ny;
wr = sqrt(wx.^2+wy.^2); % radial frequency

iiband = (abs(wr)>=fcutoff); % indices to scale like 1./f^pow

% Create fourier-domain filter
Fhat = ones(ny,nx);
Fhat(iiband) = (1./wr(iiband).^pow);
Fhat = Fhat./norm(Fhat(:))*(nx*ny);

Stim = ifft2(repmat(Fhat,[1,1,nreps]).*randn(ny,nx,nreps));
Stim = real(Stim)+imag(Stim);  % add real and imag parts

