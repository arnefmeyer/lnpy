function HessCheck(funptr, X0, ~, varargin)
% HessCheck(funptr, X0, opts, varargin);
%
% Checks analytic Gradient and Hessian (2nd derivative matrix) of a
% function 'funptr' and compares them to the finite-differencing based
% Gradient and Hessian. 
%
% Call the same as you would call fminunc: 
% > HessCheck(@lossfunc, prs0, opts, extra params...)

% Last updated: 9/23/2011  J. Pillow

% Evaluate the function at X0
[~, JJ, HH] = feval(funptr, X0, varargin{:});

% Pick a random small vector in parameter space
tol = 1e-6;  % Size of finite differencing step
rr = randn(length(X0),1).*tol;  % epsilon vector

% Evaluate at two surrounding points
[f1, JJ1] = feval(funptr, X0-rr/2, varargin{:});
[f2, JJ2] = feval(funptr, X0+rr/2, varargin{:});

% Print results
fprintf('Derivs: Analytic vs. Finite Diff = [%.4e, %.4e]\n', dot(rr, JJ), f2-f1);
fprintf('Hess: Analytic vs. Finite Diff = [%.4e, %.4e]\n', sum(HH*rr), sum(JJ2-JJ1));
