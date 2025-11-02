classdef Environment
    % Environment: Minimal OTFS-ISAC playground with OFDM framing
    % NOTE: Requires your own ISFFT(dd, M, N) implementation on path.

    properties
        % Waveform parameters
        M              = 256;          % subcarriers
        N              = 16;           % symbols
        modSize        = 4;            % QAM order
        deltaf         = 15e3 * 2^4;   % subcarrier spacing (Hz)
        cpSize         = 0;        % CP samples (default M/4) — adjust after M if you change M

        % Physical constants
        c0             = physconst('LightSpeed'); % m/s (Phased Array TB)
        fc             = 30e9;         % carrier (Hz)

        % Target / channel "defaults" (you can ignore if you pass explicit paths)
        TargetList     = {};            % e.g., array of structs/classes
        targetDistance = 30;            % meters
        targetVelocity = 72/3.6;        % m/s

        % Noise
        SNRdB          = -10;
    end

    properties (Dependent, SetAccess = private)
        T                    % symbol duration = 1/deltaf
        cpDuration           % seconds
        maximumSensingRange  % meters (from CP)
    end

    methods
        %% --------- Derived getters ----------
        function v = get.T(obj)
            v = 1/obj.deltaf;
        end
        function v = get.cpDuration(obj)
            v = (obj.cpSize/obj.M) * obj.T;
        end
        function v = get.maximumSensingRange(obj)
            v = obj.c0 * obj.cpDuration / 2;
        end

        %% --------- Bit generation ----------
        function dataBits = createDataBits(obj)
            bitsPerSym = log2(obj.modSize);
            dataBits   = randi([0 1], obj.M*obj.N, bitsPerSym);
        end

        %% --------- Baseband OTFS Tx (time-domain with CP) ----------
        function [txSignal,dd] = createTxSignal(obj, dataBits)
            % Map bits -> QAM on DD grid, ISFFT to TF, OFDM to time, add CP.
            bitsPerSym = log2(obj.modSize);
            if size(dataBits,1) ~= obj.M*obj.N || size(dataBits,2) ~= bitsPerSym
                error('dataBits must be (M*N) x log2(modSize).');
            end

            % Bits -> symbols on DD grid
            dataDe  = bi2de(dataBits);               % (M*N) x 1, natural mapping
            dataDe  = reshape(dataDe, obj.M, obj.N); % M x N
            dd      = qammod(dataDe, obj.modSize, 'UnitAveragePower', true);

            % DD -> TF via ISFFT 
            tf      = ISFFT(dd, obj.M, obj.N);       % M x N (subcarrier x symbol)

            % TF -> time by per-symbol IFFT
            ofdmNoCP = ifft(tf, [], 1) * sqrt(obj.M);   % IFFT across subcarriers

            % Add CP per column
            ofdmCP = obj.insertCP(ofdmNoCP, obj.cpSize); % (M+cpSize) x N

            % Serialize to single stream
            txSignal = ofdmCP(:);
        end

        %% --------- ISAC convenience (same as Tx in this skeleton) ----------
        function txSignal = isacTransmitter(obj, dataBits)
            txSignal = obj.createTxSignal(dataBits);
        end

        %% --------- Channel realization (multi-path delay-Doppler) ----------
        function rxSignal = realizeChannel(obj, alpha, delays, dopplers, txSignal)
            % alpha    : P x 1 complex path gains
            % delays   : P x 1 (seconds)
            % dopplers : P x 1 (Hz)
            % txSignal : ((M+cpSize)*N) x 1 time-domain signal with CP

            if ~isvector(alpha) || ~isvector(delays) || ~isvector(dopplers)
                error('alpha, delays, dopplers must be vectors of same length.');
            end
            alpha    = alpha(:);
            delays   = delays(:);
            dopplers = dopplers(:);
            P        = numel(alpha);
            if ~(numel(delays)==P && numel(dopplers)==P)
                error('alpha, delays, dopplers lengths must match.');
            end

            % Time index in seconds for each baseband sample
            % Sample period = T/M (per OFDM time-domain sample)
            Ts = obj.T/obj.M;
            L  = numel(txSignal);
            tfSignal = fft(reshape(txSignal, obj.M, obj.N)) / sqrt(obj.M);

            rx = zeros(L,1);
            for p = 1:P
                % Integer-sample delay approximation (simple + robust)
                txSignal_delay = zeros(obj.M * obj.N, 1);
                l_tau = ceil(delays(p) / (obj.T / obj.M));
                txSignal_delay(:, 1) = circshift(reshape(circshift(ifft(diag(exp(-1j * 2 * pi * (0:1:(obj.M-1)) *  obj.deltaf * delays(p))) * tfSignal ) * sqrt(obj.M), - l_tau ), [], 1), l_tau);
                dopplerEffect = exp(1j * 2 * pi * dopplers(p) .* (0:1:(obj.M*obj.N - 1))' * obj.T / obj.M);
                sig = alpha(p)* dopplerEffect .* txSignal_delay;
                sig = sum(sig, 2);
                rx = rx + sig;
            end

            % Add AWGN at target SNR (per sample, UnitAveragePower symbols)
            rxSignal = obj.addAwgn(rx, obj.SNRdB);
        end
    
        function [estimatedVelocity,estimatedRange]= estimateRangeVelocity(obj,rxSignal,ddSignal)
        % Sensing receiver
        Ytf = WignerTransform(rxSignal, obj.M, obj.N);
        Ydd = SFFT(Ytf, obj.M, obj.N);
        Xdd = ddSignal;
        
        % Two-phase sensing estimation algorithm
        ydd = Ydd(:);
        K = 60;
        % phase I
        delayList = (0:1:(obj.M-1)) * obj.T / obj.M;
        DopplerList = (-obj.N/2:1:(obj.N/2 - 1)) * obj.deltaf / obj.N;
        profile = zeros(obj.M, obj.N);
        for m = 1:length(delayList)
            for n = 1:length(DopplerList)
                
                ydd_p = obj.OTFS_approximatedOutput(Xdd, obj.T, delayList(m), DopplerList(n));
                profile(m, n) = abs(ydd_p' * ydd)^2;
            end
        end
        [~, index] = max(profile(:));
        [mi, ni] = ind2sub(size(profile), index);
        % phase II
        phi = double( (sqrt(5) - 1) / 2);
        a1 = mi - 2; b1 = mi;
        a2 = ni - obj.N/2 - 2; b2 = ni - obj.N/2;
        for k = 1:K
            I1 = b1 - a1; I2 = b2 - a2;
            x1 = a1 + (1 - phi) * I1; x2 = a1 + phi * I1;
            y1 = a2 + (1 - phi) * I2; y2 = a2 + phi * I2;
            ydd_11 = obj.OTFS_output(Xdd, obj.T, x1 * obj.T / obj.M, y1 * obj.deltaf / obj.N);
            ydd_12 = obj.OTFS_output(Xdd, obj.T, x1 * obj.T / obj.M, y2 * obj.deltaf / obj.N);
            ydd_21 = obj.OTFS_output(Xdd, obj.T, x2 * obj.T / obj.M, y1 * obj.deltaf / obj.N);
            ydd_22 = obj.OTFS_output(Xdd, obj.T, x2 * obj.T / obj.M, y2 * obj.deltaf / obj.N);
            f11 = abs(ydd_11' * ydd)^2;
            f12 = abs(ydd_12' * ydd)^2;
            f21 = abs(ydd_21' * ydd)^2;
            f22 = abs(ydd_22' * ydd)^2;
            [~, fmax] = max([f11, f12, f21, f22]);
            switch fmax
                case 1, b1 = x2; b2 = y2;
                case 2, b1 = x2; a2 = y1;
                case 3, a1 = x1; b2 = y2;
                case 4, a1 = x1; a2 = y1;
            end
        end
        estimatedDelay = (a1 + b1) / 2 * obj.T / obj.M;
        estimatedDoppler = (a2 + b2) / 2 * obj.deltaf / obj.N;
        estimatedRange = estimatedDelay * obj.c0 / 2;
        estimatedVelocity = estimatedDoppler * obj.c0 / obj.fc / 2;
        % Hp = OTFS_output(Xdd, obj.T, estimatedDelay, estimatedDoppler);
        % estimatedAlpha = (Hp' * Hp) \ (Hp' * ydd);

        end

        function ydd = OTFS_approximatedOutput(obj,Xdd, T, delay, Doppler)
        [M, N] = size(Xdd);
        lt = ceil(delay / (T / M));
        deltaf = 1 / T;
        kn = ceil(Doppler / (deltaf / N));
        Ydd = circshift(Xdd, [lt kn]);
        ydd = Ydd(:);
        end



    end


    %% ---------------- Private helpers ----------------
    methods (Access = private)
        function xcp = insertCP(~, x, cp)
            % x: M x N, cp: scalar
            [M, N] = size(x);
            pre = x(M-cp+1:M, :);
            xcp  = [pre; x];
        end

        function y = addAwgn(~, x, SNRdB)
            if isinf(SNRdB)
                y = x; return;
            end
            Es  = mean(abs(x).^2);
            SNR = 10.^(SNRdB/10);
            N0  = Es / SNR;
            n   = sqrt(N0/2) * (randn(size(x)) + 1j*randn(size(x)));
            y   = x + n;
        end
    %% ---------------- SENSING FUNCTIONS ----------------


        
        
        function ydd = OTFS_output(obj,Xdd, T, delay, Doppler)
        [M, N] = size(Xdd);
        lt = ceil(delay / (T / M));
        deltaf = 1 / T;
        Xtf = ISFFT(Xdd, M, N);
        rt = exp(1j * 2 * pi * Doppler * (0:1:(M*N - 1))' * T / M) .* circshift(reshape(circshift(ifft(diag(exp(-1j * 2 * pi * (0:1:(M-1)) *  deltaf * delay)) * Xtf ) * sqrt(M), - lt ), [], 1), lt);
        Rt = reshape(rt, M, N);
        Ydd = fft(Rt.').' / sqrt(N);
        ydd = Ydd(:);
        end
        

    end

end
