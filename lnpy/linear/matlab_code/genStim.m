function Stimuli = genStim(nDims, nStim, whichStim)
%
%--------------------------------------------------------------------------
% genStim.m: generate stimuli given # of dim, stimuli, and kinds
%--------------------------------------------------------------------------
%
% INPUT ARGUMENTS
%   nDims   a vector for filter dimensions
%               e.g.) 1-d, nDims = nX
%                       2-d, nDims = [nT, nX]
%                       3-d, nDims = [nT, nY, nX]
%   nStim   number of stimuli
%   whichStim  which stimuli? 1/f or white noise
%                    1: 1/f stimuli
%                    2: white noise
%
% OUTPUT ARGUMENTS:
%  Stimuli: datastructure which has
%               xTraining      Training data (nstim* (spatialdims*nt))
%               xTest           Test data 
%               xraw_training  Training raw data (nstim* spatialdims) 
%               xraw_test       Test raw data (nstim* spatialdims) 
% 
% Updated: 25/12/2011 Mijung Park

lengDims = length(nDims);

if lengDims == 1 % 1d stimuli
    nX = nDims;
    if whichStim == 1 % 1/f stimuli
        % 1/f stimuli
        pow = 2; % to determine the power spectral density slope
        samprate = 100; % sampling rate in Hz
        fcutoff = 10/nX; % low-frequency cutoff
        Stimuli.xTraining = make1Fstim_1D([nX nStim],samprate,fcutoff,pow)'; % transpose because each column has 1/f stimuli
        Stimuli.xTest = make1Fstim_1D([nX nStim],samprate,fcutoff,pow)';
       
    else % whichStim==2
        %  white noise stimuli
        Stimuli.xTraining = randn(nStim, nX);
        Stimuli.xTest = randn(nStim, nX);
        
    end
    
    
elseif lengDims == 2 % 2d stimuli
    nT = nDims(1);
    nX = nDims(2);
    if whichStim == 1 % 1/f stimuli
        % 1/f stimuli
        flo = 1/nT;   % Low-pass cutoff (Hz)  (Should be >= 1./min(nx,nt))
        pow = 2;    % Exponent for falloff with frquency (must be positive)        
        Stim = make1Fstim_2D([nT,nX],flo,pow,nStim);
        Stimuli.xTraining = reshape(Stim,nX*nT, nStim)'; % Training Data    
        Stim_tst = make1Fstim_2D([nT,nX],flo,pow,nStim);
        Stimuli.xTest  = reshape(Stim_tst, nX*nT, nStim)'; % Test data

    else % whichStim == 2
        % white noise stimuli
        Stim = randn(nX,nT,nStim);
        Stimuli.xTraining = reshape(Stim, nX*nT, nStim)';       
        Stim_tst = randn(nX,nT,nStim);
        Stimuli.xTest = reshape(Stim_tst, nX*nT, nStim)';
        
    end
    
else % lengDims == 3
    nT = nDims(1);
    nY = nDims(2);
    nX = nDims(3);
    
    % define a spatial filter
    if whichStim == 1 % 1/f stimuli
        % 1/f stimuli
        flo = 1/nY;   % Low-pass cutoff (Hz)  (Should be >= 1./min(nx,ny))
        pow = 2;    % Exponent for falloff with frquency (must be positive)
       
        Stim = make1Fstim_2D([nY,nX],flo,pow,nStim); % Training Data          
        Stim_tst = make1Fstim_2D([nY,nX],flo,pow,nStim); % Test data
        
    else % whichStim == 2
        % white noise stimuli
        Stim = randn(nX,nY,nStim);
        Stim = reshape(Stim, nX*nY, nStim)';       
        Stim_tst = randn(nX,nY,nStim);
        Stim_tst = reshape(Stim_tst, nX*nY, nStim)';
    end
        
    % Then filter the spatial filter with a temporal filter
    Corr_Stim= filter(exp(0:-1/6:-nT+1),1, Stim, [], 3);
    permStim = permute(Corr_Stim, [3 1 2]);
    Stimuli.xraw_training = reshape(permStim, nStim, []);
    Stimuli.xTraining = makeStimRows(Stimuli.xraw_training, nT);
    
    Corr_Stim_tst= filter(exp(0:-1:-nT+1),1,Stim_tst, [], 3); % filter with a temporal kernel
    permStimtst = permute(Corr_Stim_tst, [3 1 2]); % permute for temporal dimension to be first
    Stimuli.xraw_test = reshape(permStimtst, nStim, []);
    Stimuli.xTest = makeStimRows(Stimuli.xraw_test, nT);

end
        