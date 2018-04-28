function X = toeplitz2D(im,ny,nx);
%  X = toeplitz2D(im,ny,nx);
%
%  Takes an image 'im' and creates a stack of images with every possible
%  horizontal and vertical shift of the image (circular boundary
%  conditions).  Images passed back as column vectors of a matrix.

if nargin <= 1
    [ny,nx] = size(im);
end
[ni, nj] = size(im);

X = zeros(ny,nx,ny*nx);

yinds = [1:ny,1:ni];
xinds = [1:nx,1:nj];

for i = 1:nx
    for j = 1:ny
        X(yinds(j:j+ni-1),xinds(i:i+nj-1),ny*(i-1)+j) = im;
    end
end
X = reshape(X,ny*nx,ny*nx);    
