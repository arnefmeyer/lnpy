function [khat,logEv,prs] = runALDf_paramM(prs0, datastruct, opts)

ndims = datastruct.ndims;
leng_ndims = length(ndims);

% set bounds on estimated parameters
%     [nsevar; mu_x; mu_y; sig_rad; sig_orth; oscale];
noiseRange = [1e-3, 1e3];
m = [-1e3, 1e3];
mu = [-ndims, ndims];
oscaleRange = [-50, 50];

if leng_ndims ==1
    LB = [noiseRange(1); m(1); mu(1);oscaleRange(1)]; 
    UB = [noiseRange(2); m(2); mu(2);oscaleRange(2)];
elseif leng_ndims ==2
    LB = [noiseRange(1); m(1); m(1); m(1); mu(:,1); oscaleRange(1)]; 
    UB = [noiseRange(2); m(2); m(2); m(2); mu(:,2); oscaleRange(2)];
else % 3d
    df = 1; 
    mu_x = [-ndims(1)/2-df, ndims(1)/2+df];
    mu_y = [-ndims(2)/2-df, ndims(2)/2+df];
    mu_t = [-ndims(3)/2-df, ndims(3)/2+df];
    gam = [-1e-3, 1e-3];
    LB = [noiseRange(1); mu_t(1); mu_y(1) ; mu_x(1); sig_rad(1); sig_orth(1); sig_orth(1); gam(1); oscaleRange(1)]; 
    UB = [noiseRange(2); mu_t(2); mu_y(2); mu_x(2); sig_rad(2); sig_orth(2); sig_orth(2); gam(2); oscaleRange(2)];    
end

lfun = @(p)negLogEv_ALDf_paramM(p,datastruct);

% ------ Optimize evidence --------------------------------------
[prs, fval, exitflag, output, lambda, grad, hessian] = fmincon(lfun,prs0,[],[],[],[],LB,UB,[],opts);

% ------ compute filter and posterior variance at maximizer --------------
% [logEv,kh,Cinv,Crnk] = lfun(prs);
[logEv,kh,Cinv,Crnk] = lfun(prs);

if leng_ndims==1
    M = FFTmatrix(ndims);
    khat_cc = M'*kh;
    khat = real(khat_cc) + imag(khat_cc);
elseif leng_ndims==2
    nt = ndims(1);
    nx = ndims(2);
    M = FFT2matrix(nt, nx);
    khat_cc = reshape(M'*kh, nt, nx);
    khat = real(khat_cc)+ imag(khat_cc);
else % if k is 3 dim.
    nt = ndims(1);
    ny = ndims(2);
    nx = ndims(3);
    BB = FFT3matrix(nt, ny, nx);
    khat_cc = permute(reshape(BB'*kh(:), nt, ny, nx), [2 3 1]);
    khat = real(khat_cc) + imag(khat_cc);
end

fprintf('ALDf: terminated with rank(Cprior)=%d\n',Crnk);
