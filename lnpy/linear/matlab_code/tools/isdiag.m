function y = isdiag(A)
% y = issym(A)
%
% determine if A is diagonal

if issparse(A)
    y = isequal(A,spdiags(diag(A),0,size(A,1)));
else
    y = isequal(A,diag(diag(A)));
end
