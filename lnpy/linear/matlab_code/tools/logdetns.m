function x = logdetns(A)
% LOGDET - computes the log-determinant of a matrix A using Cholesky or LU
% factorization
%
% LOGDET
%
% x = logdet(A);
%
% This is faster and more stable than using log(det(A))
%
% Input:
%     A NxN - A must be sqaure, positive SYMMETRIC and semi-definite
%     (Chol assumes A is symmetric)

if checksym(A)  % crude symmetry check

    x = 2*sum(log(diag(chol(A))));

else
    
    x = sum(log(abs(diag(lu(A)))));

end

% ----------------------------
function z = checksym(A)
% Crude (but fast) algorithm for checking symmetry, just by looking at the
% first column and row

z = isequal(A(:,1),A(1,:)');