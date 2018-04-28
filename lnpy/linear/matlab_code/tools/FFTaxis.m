function [fx] =  FFTaxis(nx)

% [fx] =  FFTaxis(nx)
%
% Computes the frequency coordinates underlying the 2D FFT on a signal of
% size [nx], fx = [0:floor((nx)/2), -floor((nx-1)/2):-1];
 
fx = [0:floor((nx)/2), -floor((nx-1)/2):-1];

    