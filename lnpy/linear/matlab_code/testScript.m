%% Testscript for testing ALDs, ALDf, and ALDsf for 1D, 2D, & 3D stimuli 

% 1D, 2D, and 3D simulated example code:
% 
% Important variables & functions:
%   nstim: # of stimuli
%   nsevar: noise variance   
%   nkt  = number of time samples of x (stimulus) to use to predict y (response).
%   filterdims = dimension of input stimulus
%        e.g.) 1D: nx (if nkt=1) 
%              2D: [nkt; nx] or [ny; nx]
%              3D: [nkt; ny; nx]
%   whichstim: choose either white noise or correlated stimuli
%
% genkTrue.m generates true RF by the difference of two Gaussians
% genStim.m generates chosen stimuli 
% runALD.m provides RF estimates using empirical Bayes with ALDs,f,sf priors
%
% Updated: 17/01/2012 by Mijung Park & Jonathan Pillow

%% Add a path to use functions in the 'tools' folder

addpath('tools/');

%% 1D stimuli example

clear;
clc;

nstim = 1000; % number of stimuli
nsevar = 1; %  noise variance

% true filter with spatial dimension nx, difference of two Gaussians.
filterdims = 100;
ktrue = genkTrue(filterdims); % ktrue in 1d
% plot(ktrue);

% 1. generate stimuli (choose either 1/f stimuli or white noise stimuli)
whichstim = 2; % white noise stimuli, if you want 1/f stimuli, set "whichstim=1"
Stimuli = genStim(filterdims, nstim, whichstim);
 
% 2. generate noisy response: training & test data
ytraining = Stimuli.xTraining*ktrue + randn(nstim,1)*nsevar; % training data
% ytest = Stimuli.xTest*ktrue + randn(nstim,1)*nsevar; % test data (for cross-validation)

% 3. ALDs,f,sf, ML, and ridge regression
nkt = 1; 
[khatALD kridge] = runALD(Stimuli.xTraining, ytraining, filterdims, nkt);

figure(1); 
plot([ktrue, kridge, khatALD.khatSF]);
legend('true', 'ridge', 'ALDsf');

%% 2D stimuli example

clear;
clc;

nstim = 2500; % number of samples
nsevar = 1; % noise variance

% true filter (2d-Gabor) with length ny by nx
ny = 18;
nx = 18; 
filterdims = [ny; nx]; % spatial dimensions
ktrue = genkTrue(filterdims); % ktrue in 2d
% plot(ktrue(:)); imagesc(ktrue); colormap gray ; axis xy

RF_reshaped = reshape(ktrue, [], 1);

% 1. generate stimuli (choose either 1/f stimuli or white noise stimuli)
whichstim = 2; % white noise stimuli, if you want 1/f stimuli, set "whichstim=1"
Stimuli = genStim(filterdims, nstim, whichstim);

% 2. generate noisy response: training & test data
ytraining = Stimuli.xTraining*RF_reshaped + randn(nstim,1)*nsevar; % training data
% ytest = Stimuli.xTest*RF_reshaped + randn(nstim,1)*nsevar; % test data

% 3. ALDs,f,sf, ML, and ridge regression
nkt=1;
[khatALD kridge] = runALD(Stimuli.xTraining, ytraining, filterdims, nkt);

figure(2);
subplot(131);imagesc(ktrue); colormap gray; axis image; title('true');
subplot(132);imagesc(reshape(kridge, ny, nx)); axis image; title('ridge');
subplot(133);imagesc(reshape(khatALD.khatSF, ny, nx)); axis image; title('ALDsf');

%% 3D stimuli example

clear;
clc;

nstim = 2000; % number of samples
nsevar = 0.5; % noise variance

% true filter (3d-Gabor) with length nt by ny by nx
% Note: if total dimensionality is too high, this code becomes slow (our future work will solve this issue).        
nkt = 8;
ny = 6;
nx = 6; 

spatialdims = [ny; nx]; % spatial dimension of input stimulus
filterdims = [nkt; spatialdims]; % filter dimensions
ktrue = genkTrue(filterdims); % ktrue in 3d

% plotting true k
% for i=1:nkt
%     imagesc(ktrue(:,:,i)); colormap gray; axis image;
%     pause(0.2);
% end

ktrue_perm = permute(ktrue, [3 1 2]); % time axis is the first

RF_reshaped = reshape(ktrue_perm, [], 1); % reshape to a vector

% 1. generate stimuli (choose either 1/f stimuli or white noise stimuli)
whichstim = 2; %  white noise stimuli, if you want 1/f stimuli, set "whichstim=1"
Stimuli = genStim(filterdims, nstim, whichstim);

% 2. generate noisy response: training & test data
ytraining = Stimuli.xTraining*RF_reshaped + randn(nstim,1)*nsevar; % training data
% ytest = Stimuli.xTest*RF_reshaped + randn(nstim,1)*nsevar; % test data

% 3. ALDs,f,sf, ML, and ridge regression
[khatALD kRidge] = runALD(Stimuli.xraw_training, ytraining, spatialdims, nkt);

% 4. plot ALDsf estimate
kRidge_rsh = permute(reshape(kRidge, [nkt, spatialdims(1), spatialdims(2)]), [2 3 1]);
% kALD_rsh = permute(reshape(khatALD.khatS, [nkt, spatialdims(1), spatialdims(2)]), [2 3 1]);
% kALD_rsh = khatALD.khatF; % ALDf is already rearranged in time and space
kALD_rsh = permute(reshape(khatALD.khatSF, [nkt, spatialdims(1), spatialdims(2)]), [2 3 1]);

figure(3);
for i=1:nkt
    subplot(231); imagesc(ktrue(:,:,i)); colormap gray; axis image; title('true')
    subplot(232); imagesc(kRidge_rsh(:,:,i)); colormap gray; axis image; title('ridge')
    subplot(233); imagesc(kALD_rsh(:,:,i)); colormap gray; axis image; title('ALDsf')
    subplot(234); plot(ktrue(:,:,i)); title('true');
    subplot(235); plot(kRidge_rsh(:,:,i)); title('ridge');
    subplot(236); plot(kALD_rsh(:,:,i)); title('ALDsf');
    
    pause(1);
end
