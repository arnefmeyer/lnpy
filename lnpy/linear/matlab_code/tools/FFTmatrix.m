function M = FFTmatrix(n)
%  M = FFTmatrix(n)
%  Computes the matrix that performs the DFT on a vector of length(n);
%  i.e.  FFTmatrix(n)*vec  is the same as fft(vec).

M = fft(eye(n));

