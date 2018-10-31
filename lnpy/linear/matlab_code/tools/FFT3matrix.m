function M = FFT3matrix(m,n, l)
%  M = FFT3matrix(m,n,l)
% 
% Computes the matrix that performs the fft3 on an image of size m x n x l.
%
% i.e:    fft3(A) = reshape(FFT3matrix(m,n, l)*A(:), m, n, l)

RawMat = reshape(eye(m*n*l), m, n, l, m*n*l);

M = zeros(m*n*l, m*n*l);
nrmtrm  = sqrt(m*n*l);

for i=1:m*n*l
    M(:,i) = reshape(fftn(RawMat(:,:,:,i))./nrmtrm, [], 1);
end
