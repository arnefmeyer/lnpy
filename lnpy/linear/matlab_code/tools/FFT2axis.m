function [fy, fx] =  FFT2axis(ny, nx)
% [fy, fx] =  FFT2axis(ny, nx)
%
% Computes the frequency coordinates underlying the 2D FFT on a signal of
% size [ny, nx], wx = [0:floor((nx)/2), -floor((nx-1)/2):-1];

wx = [0:floor((nx)/2), -floor((nx-1)/2):-1];
wy = [0:floor((ny)/2), -floor((ny-1)/2):-1];
[fy, fx] = ndgrid(wy, wx);

% modify highest frequency vals for even-numbered grids
if mod(ny,2)==0
    ctrrow = ny/2+1;
    fy(ctrrow,ceil(nx/2+1):end) = -fy(ctrrow,ceil(nx/2+1):end);
end
if mod(nx,2)==0
    ctrcol = nx/2+1;
    fx(ceil(ny/2+1):end,ctrcol) = -fx(ceil(ny/2+1):end,ctrcol);
end
    