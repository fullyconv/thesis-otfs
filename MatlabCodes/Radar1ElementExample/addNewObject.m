function [rxSignal] = addNewObject(alpha,targetDelay,targetDoppler,txSignal )


delay = targetDelay;
doppler = targetDoppler;
tfSignal = fft(reshape(txSignal, M, N)) / sqrt(M);
txSignal_delay = zeros(M * N, 1);
l_tau = ceil(delay / (T / M));
txSignal_delay(:, 1) = circshift(reshape(circshift(ifft(diag(exp(-1j * 2 * pi * (0:1:(M-1)) *  deltaf * delay)) * tfSignal ) * sqrt(M), - l_tau ), [], 1), l_tau);
dopplerEffect = exp(1j * 2 * pi * doppler .* (0:1:(M*N - 1))' * T / M);
rxSignal = repmat(alpha, M*N, 1) .* dopplerEffect .* txSignal_delay;
rxSignal = sum(rxSignal, 2);

end