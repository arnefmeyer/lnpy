function M = FFT2matrix(m,n)
%  M = FFT2matrix(m,n)
% 
% Computes the matrix that performs the fft2 on an image of size m x n.
%
% i.e:    fft2(A) = reshape(FFT2matrix(m,n)*A(:), m, n)

RawMat = reshape(eye(m*n),m,n,m*n);
M = reshape(fft2(RawMat),m*n,m*n);