        function [Stim,freqs,filt] = make1Fstim_1D(nsamps,samprate,fcutoff,pow)
        % [Stim,freqs,filt] = make1Fstim_1D(nsamps,samprate,fcutoff,pow);
        %
        % Create a 1D Gaussian stimulus of length nx, with power spectrum that
        % falls off as 1./f.^pow;
        %
        % Inputs: 
        %   nsamps = number of samples desired
        %          = [nsamps, nrepetitions], if 2-vector passed in
        %   samprate = assumed sampling rate (Hz)
        %   fcutoff = low-frequency cutoff (stimulus will be white below this)
        %   pow = assumed exponent for fall-off of frequencies
        %
        % Outputs:
        %   Stim = 1/F stimulus.  (Each column is a single 1/F stimulus)
        %   freqs = vector of frequencies used when computing 1/|F|.^pow
        %   filt = equivalent filter, which could be used to generate stimuli via
        %          circonv:  e.g., Stim = circonv(randn(nsamps,1),filt));
        %   
        % Example call:
        % % create 20000-sample stimulus with 1/F spectrum above .1 Hz
        % > Stim = make1Fstim_1D(20000,1000,.1,1); 

        % Process input args
        if length(nsamps)==1
            nreps = 1;
        else
            nreps = nsamps(2);
            nsamps = nsamps(1);
        end

        nsecs = nsamps/samprate;  % Number of seconds in stimulus

        % Set up bins for time and frequency
        xs = (1:nsamps)';  % bin indices
        freqs = mod(xs+nsamps/2-1,nsamps)-nsamps/2; % frequency bins
        freqs = freqs/nsecs;

        iiband = (abs(freqs)>fcutoff); % indices to scale like 1./f^pow

        % Create filter in Fourier-domain
        fhat = ones(nsamps,1);
        fhat(iiband) = 1./abs(freqs(iiband)).^pow;
        fnrm = norm(fhat);  
        fhat = fhat./fnrm*(nsamps);  % normalize filter so marginal pixel dist is 1

        Stim = ifft(repmat(fhat,1,nreps).*randn(nsamps,nreps));  % Create & filter in Fourier domain
        Stim = real(Stim)+imag(Stim);  % add real and imag parts

        if nargout > 2
            filt = ifft(fhat);
            filt = fftshift(filt./norm(filt));
        end
