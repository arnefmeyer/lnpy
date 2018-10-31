function [ft, fy, fx] =  FFT3axis(nt, ny, nx)
% [ft, fy, fx] =  FFT3axis(nt, ny, nx)
%
% Computes the frequency coordinates underlying the 3D FFT on a signal of
% tensor size [nt, ny, nx]
 
wt = [0:floor((nt)/2), -floor((nt-1)/2):-1];
wy = [0:floor((ny)/2), -floor((ny-1)/2):-1];
wx = [0:floor((nx)/2), -floor((nx-1)/2):-1];
[ft, fy, fx] = ndgrid(wt, wy, wx);

% modify highest frequency vals for even-numbered grids
if mod(nt,2)==0
    ctrrow = nt/2+1;
    ft(ctrrow,:,:) = ft(ctrrow,:,:).*compsign(fy(ctrrow,:,:),fx(ctrrow,:,:));
end
if mod(ny,2)==0
    ctrcol = ny/2+1;
    fy(:,ctrcol,:) = fy(:,ctrcol,:).*compsign(ft(:,ctrcol,:),fx(:,ctrcol,:));
end
if mod(nx,2)==0
    ctrplt = nx/2+1;
    fx(:,:,ctrplt) = fx(:,:,ctrplt).*compsign(ft(:,:,ctrplt),fy(:,:,ctrplt));
end


function ss = compsign(aa,bb)
% Compute the desired sign

ss = ones(size(aa));
ss(aa~=0) = sign(aa(aa~=0));
ss((aa==0)&(bb~=0))= sign(bb((aa==0)&(bb~=0)));
