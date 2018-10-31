function [GInit] = compGriddedLE(fptr, InitialValues, datastruct)

%--------------------------------------------------------------------------
% compGriddedLE.m: find a good initial point for evidence optimization
%--------------------------------------------------------------------------
%
% DESCRIPTION:
%    computes evidence on the grid of points
%    to find the best initial point for optimization
%
% INPUT ARGUMENTS:
%    fptr - objective function (evidence)
%    InitialValues - grid points
%    datastruct - has  XX, XY, YY, stimulus dimension
%
% OUTPUT ARGUMENTS:
%    GInit - initial values for hyperparameters that has the max evidence
%
% (Updated: 25/12/2011 Mijung Park) 

HowManyInit = size(InitialValues, 1);
negLogEvid = zeros(HowManyInit,1);

for i=1:HowManyInit 
    prs_p = InitialValues(i,:)';
    negLogEvid(i) = fptr(prs_p, datastruct);
end

minLogevid =(min(negLogEvid)); % find a minimum neg log evid and its index
indx = (negLogEvid == minLogevid);
GInit = InitialValues(indx,:)'; % choose the best initial value for optimization
GInit = GInit(:,1); % choose first one if there are more than one minimium value