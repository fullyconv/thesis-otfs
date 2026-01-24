% ESPRIT Algorithm for DOA Estimation
clear; clc;

%% 1. Signal Simulation Parameters
N = 10;                 % Number of sensors (ULA)
M = 2;                  % Number of source signals
K = 1000;               % Number of snapshots (time samples)
d = 0.5;                % Sensor spacing (in wavelengths)
theta = [-15, 20];      % True angles of arrival (in degrees)
snr = -10;               % Signal-to-Noise Ratio (dB)

% Convert angles to radians
theta_rad = deg2rad(theta);

% Steering matrix A (N x M)
% A(n,m) = exp(-j * 2 * pi * d * (n-1) * sin(theta_m))
A = exp(-1i * 2 * pi * d * (0:N-1)' * sin(theta_rad));

% Generate source signals (Random Gaussian)
S = (randn(M, K) + 1i * randn(M, K)) / sqrt(2);

% Additive White Gaussian Noise
X = A * S;
X = awgn(X, snr, 'measured');

%% 2. ESPRIT Algorithm
% Step 1: Compute Covariance Matrix
R = (X * X') / K;

% Step 2: Eigenvalue Decomposition
[V, D] = eig(R);
[~, idx] = sort(diag(D), 'descend');
V = V(:, idx);

% Step 3: Extract Signal Subspace (Us)
Us = V(:, 1:M);

% Step 4: Split the array into two overlapping subarrays
% Subarray 1: Sensors 1 to N-1
% Subarray 2: Sensors 2 to N
Us1 = Us(1:N-1, :);
Us2 = Us(2:N, :);

% Step 5: Solve for the Rotation Matrix (Phi)
% Using Total Least Squares (TLS) or Simple Least Squares
Phi = Us1 \ Us2; 

% Step 6: Find Eigenvalues of Phi
eigenvals = eig(Phi);

% Step 7: Estimate Angles
% The eigenvalues correspond to exp(-j * 2 * pi * d * sin(theta))
estimated_theta = asind(-angle(eigenvals) / (2 * pi * d));

%% 3. Display Results
fprintf('True Angles:      %s degrees\n', num2str(theta));
fprintf('Estimated Angles: %s degrees\n', num2str(sort(estimated_theta)'));