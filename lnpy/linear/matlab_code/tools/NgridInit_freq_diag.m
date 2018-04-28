function InitialValues = NgridInit_freq_diag(ndims, nsevar, ovsc)
%--------------------------------------------------------------------------
% NgridInit_freq_diag: find good initial point for evidence optimization
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
    
    % ngrid for finding initial values
    a =nx./(nx*(2.^[0:ceil(log(nx))]));
    leng_gridf = length(a(:));
    InitialValues = [nsevar*ones(leng_gridf,1), a(:), 0.5*ovsc*ones(leng_gridf,1)];
    
elseif numb_dims ==2       
    ny = ndims(1);
    nx = ndims(2);
    
    % ngrid for finding initial values
    [m1, m2] = ndgrid(nx./(nx*(2.^[0:ceil(log(nx))])), ny./(ny*(2.^[0:ceil(log(ny))])));
    leng_gridf = length(m1(:));
    InitialValues = [nsevar*ones(leng_gridf,1), m1(:), m2(:), 0.5*ovsc*ones(leng_gridf,1)];
    
else % numb_dims ==3
    nx = ndims(1);
    ny = ndims(2);
    nt = ndims(3);

    % ngrid for finding initial values
    [m1, m2, m3] = ndgrid(nt./(nt*(2.^[0:floor(log(nt))])), ny./(ny*(2.^[0:floor(log(ny))])), nx./(nx*(2.^[0:ceil(log(nx))])));
    leng_gridf = length(m1(:));
    InitialValues = [nsevar*ones(leng_gridf,1), m1(:), m2(:), m3(:), 0.5*ovsc*ones(leng_gridf,1)];
end

