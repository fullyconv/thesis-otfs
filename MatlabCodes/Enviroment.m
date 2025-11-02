classdef Enviroment

    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here

    properties
        % Waveform parameters
        M = 256; % subcarrier number
        N = 16; % symbol number
        modSize = 4; % modulation size
        deltaf = 15e3 * 2^4; % subcarrier spacing
        T = 1 / deltaf; % symbol duration
        cpSize = M / 4;
        cpDuration = cpSize / M * T;
        c0 = physconst('LightSpeed'); % light of speed
        fc = 30e9; % carrier frequency
%TARGET PARAMETERS
        TargetList=Platform

        targetDistance = 30;
        targetVelocity = 72 / 3.6;

        SNRdB = -10;
        maximumSensingRange = c0 * cpDuration / 2;

    end

    methods(Static)
        function [dataBits]=createDataBits()
        dataBits = randi([0 1], obj.M * obj.N, log2(obj.modSize));
        end
    end
    
    methods
        

        function [txSignal]=createTxSignal(dataBits)
            dataDe = bi2de(dataBits);
            dataDe = reshape(dataDe, obj.M, obj.N);
            data = qammod(dataDe, obj.modSize, 'UnitAveragePower',true);
            ddSignal = data;
            tfSignal = ISFFT(ddSignal, obj.M, obj.N);
            txFrame = ifft(tfSignal) * sqrt(obj.M);
            txSignal = reshape(txFrame, [], 1);
        end


        % function [transmittedSignal]=Transmit()
        % 
        %     for obj_idx=1:length(obj.TargetList)
        %         obj.TargetList().
        %         targetDelay = range2time(targetDistance, c0);
        %         targetDoppler = speed2dop(2 * targetVelocity, c0 / fc);
        %         targetCoefficient = exp(1j * 2 * pi * rand());
        % 
        %     end
        % 
        % end

        function [rxSignal] = realizeChannel(alpha,targetDelay,targetDoppler,txSignal )
        delay = targetDelay;
        doppler = targetDoppler;
        tfSignal = fft(reshape(txSignal, obj.M, obj.N)) / sqrt(obj.M);
        txSignal_delay = zeros(obj.M * obj.N, 1);
        l_tau = ceil(delay / (obj.T / obj.M));
        txSignal_delay(:, 1) = circshift(reshape(circshift(ifft(diag(exp(-1j * 2 * pi * (0:1:(obj.M-1)) *  obj.deltaf * delay)) * tfSignal ) * sqrt(obj.M), - l_tau ), [], 1), l_tau);
        dopplerEffect = exp(1j * 2 * pi * doppler .* (0:1:(obj.M*obj.N - 1))' * obj.T / obj.M);
        rxSignal = repmat(alpha, obj.M*obj.N, 1) .* dopplerEffect .* txSignal_delay;
        rxSignal = sum(rxSignal, 2);       
        end


        function [txSignal]=isacTransmitter(dataBits)
        % OTFS ISAC transmitter
        
        dataDe = bi2de(dataBits);
        dataDe = reshape(dataDe, obj.M, obj.N);
        data = qammod(dataDe, obj.modSize, 'UnitAveragePower',true);
        ddSignal = data;
        tfSignal = ISFFT(ddSignal, obj.M, obj.N);
        txFrame = ifft(tfSignal) * sqrt(obj.M);
        txSignal = reshape(txFrame, [], 1);
        end


    end
end