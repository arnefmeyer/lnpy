function dx = finitediff(x,dt)
% dx = finitediff(x,dt);
%
% Compute a centered estimate of the derivative dx/dt by averaging forward
% and backward derivatives.

if nargin == 1
    dt = 1;
end

[m,n] = size(x);
if m==1  % Check if row vector passed in
    x = x';
    isrow = 1;
else
    isrow = 0;
end

dx1 = diff(x);
dx = .5*([dx1(1,:);dx1]+[dx1;dx1(end,:)])/dt;

if isrow  % Convert back to row vector, if row passed in
    dx = dx';
end
