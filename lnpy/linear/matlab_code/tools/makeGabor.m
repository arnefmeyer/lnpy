function f = makeGabor(ori,sf,phi,mu,stdevs,xx,yy);
%  f = makeGabor(ori,sf,phi,mu,stdevs,xx,yy);
%
%  Create a 2D Gabor function.
%
%  Inputs:
%    ori = orientation of elongated (major) axis (in radians)
%    sf = spatial frequency  (cycles / pixel)
%    phi = phase (0 = sine wave; pi/2 = cosine wave
%    mu = [mu_x, mu_y]; mean of Gaussian window
%    stdevs = [sig1, sig2]; std dev along major and minor axis
%    xx, yy = spatial coordinates.  Should be zero-centered


xrot = (xx-mu(1))*cos(ori) + (yy-mu(2))*sin(ori);
yrot = -(xx-mu(1))*sin(ori) + (yy-mu(2))*cos(ori);

% Make Elliptical Gaussian Window
GaussianWindow = exp(-.5*((xrot/stdevs(1)).^2+(yrot/stdevs(2)).^2));

% Make sine wave oscillation
SinWav = sin(2*pi*sf*yrot - phi);

f = GaussianWindow.*SinWav;