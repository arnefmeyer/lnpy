function InitialValues = NgridInit_pixel(ndims, nsevar, ovsc, kRidge)
%--------------------------------------------------------------------------
% NgridInit_pixel.m: find good initial point for evidence optimization
%--------------------------------------------------------------------------
%
% DESCRIPTION:
%    Make a coarse grid of points in hyperparameter space
%
% INPUT ARGUMENTS:
%    ndims: stimuli dimension
%      for example, 1d) ndims =[nx];
%      2d) ndims =[ny;nx];
%      3d) ndims =[nt;ny;nt];
%    ovsc: overall scale
%    nsevar: noise variance
%
% OUTPUT ARGUMENTS
%    InitialValues in ndgrid
%
% (Updated: 25/12/2011 Mijung Park) 


numb_dims = length(ndims);

if numb_dims ==1
    
    nx = ndims(1);
    
    Xind = (kRidge==max(kRidge));
    x1 = 1:nx;
    mean_x = x1(Xind);
        
    % ngrid for finding initial values
    a =nx./(2.^[1:ceil(log(nx))]);
    [gam_x, ov] = ndgrid(a, [0.5*log(ovsc) -log(ovsc)]);
    leng_grid = length(gam_x(:));
    InitialValues = [nsevar*ones(leng_grid, 1), mean_x*ones(leng_grid,1), gam_x(:), ov(:)];
    
elseif numb_dims ==2
    
    ny = ndims(1);
    nx = ndims(2);
    
    kRidge_yx = reshape(kRidge, ny, nx);
    Xind = (kRidge_yx==max(kRidge_yx(:)));
    [y1, x1] = ndgrid(1:ny,1:nx);
    mean_y = y1(Xind);
    mean_x = x1(Xind);
        
    % ngrid for finding initial values
    a1 = ny./(2.^[1:ceil(log(nx))]);
    a2 = nx./(2.^[1:ceil(log(ny))]);
    [gam_y, gam_x, ov] = ndgrid(a1, a2, [0.5*log(ovsc) -log(ovsc)]);
    phi = 0.5; 
    leng_grid = length(gam_y(:));
    InitialValues = [nsevar*ones(leng_grid, 1), mean_y*ones(leng_grid, 1), mean_x*ones(leng_grid,1), gam_y(:), gam_x(:), phi.*ones(leng_grid,1), ov(:)];
    
else % numb_dims ==3
    
    nkt = ndims(1);
    ny = ndims(2);
    nx = ndims(3);
    
    kRidge_tyx = reshape(kRidge, nkt, ny, nx);
    Xind = (kRidge_tyx==max(kRidge_tyx(:)));
    [t1, y1, x1] = ndgrid(1:nkt,1:ny,1:nx);
    mean_t = t1(Xind); 
    mean_y = y1(Xind); 
    mean_x = x1(Xind);

    [ti1, yi1, xi1, gam_t, gam_y, gam_x, phi1, phi2, phi3, ov] = ndgrid(mean_t, mean_y, mean_x, [nkt/3 nkt/4], [ny/10 ny/5], [nx/10 nx/5], 0.1, 0.1, 0.1, [0.5*log(ovsc) -log(ovsc)]);
    leng_grid = length(xi1(:)); 
    InitialValues = [nsevar*ones(leng_grid,1), ti1(:), yi1(:), xi1(:), gam_t(:), gam_y(:), gam_x(:), phi1(:), phi2(:), phi3(:), ov(:)];
end
