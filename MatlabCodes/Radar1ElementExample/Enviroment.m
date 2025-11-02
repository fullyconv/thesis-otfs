classdef Environment
    % Environment: Minimal OTFS-ISAC playground with OFDM framing
    % NOTE: Requires your own ISFFT(dd, M, N) implementation on path.

    properties
        % Waveform parameters
        M              = 256;          % subcarriers
        N              = 16;           % symbols
        modSize        = 4;            % QAM order
        deltaf         = 15e3 * 2^4;   % subcarrier spacing (Hz)
        cpSize         = 256/4;        % CP samples (default M/4) — adjust after M if you change M

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
        function txSignal = createTxSignal(obj, dataBits)
            % Map bits -> QAM on DD grid, ISFFT to TF, OFDM to time, add CP.
            bitsPerSym = log2(obj.modSize);
            if size(dataBits,1) ~= obj.M*obj.N || size(dataBits,2) ~= bitsPerSym
                error('dataBits must be (M*N) x log2(modSize).');
            end

            % Bits -> symbols on DD grid
            dataDe  = bi2de(dataBits);               % (M*N) x 1, natural mapping
            dataDe  = reshape(dataDe, obj.M, obj.N); % M x N
            dd      = qammod(dataDe, obj.modSize, 'UnitAveragePower', true);

            % DD -> TF via your ISFFT (must exist on path)
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
            t  = (0:L-1).' * Ts;

            rx = zeros(L,1);
            for p = 1:P
                % Integer-sample delay approximation (simple + robust)
                l = round(delays(p)/Ts);
                sig = circshift(txSignal, l);

                % Apply Doppler (Hz) as complex tone
                sig = sig .* exp(1j*2*pi*dopplers(p)*t);

                rx = rx + alpha(p)*sig;
            end

            % Add AWGN at target SNR (per sample, UnitAveragePower symbols)
            rxSignal = obj.addAwgn(rx, obj.SNRdB);
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
    end
end
