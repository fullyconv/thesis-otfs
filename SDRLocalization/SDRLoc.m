% Sensor Network Localization via SDR
clear; clc;

% 1. Setup: 3 Anchors in 2D
anchors = [0, 0; 10, 0; 5, 8]'; % 2x3 matrix
n_anchors = size(anchors, 2);
n_sensors = 1;

% True sensor position (for validation)
true_sensor = [5, 4]';

% Noisy distance measurements (from sensor to anchors)
% dist^2 = ||sensor - anchor||^2
d_sq = sum((repmat(true_sensor, 1, n_anchors) - anchors).^2, 1) + randn(1,3)*0.1;

% 2. Solve via CVX
cvx_begin sdp
    % We define the 'Lifted' matrix Z (size 2+n x 2+n)
    variable Z(2 + n_sensors, 2 + n_sensors) symmetric
    
    % Define the sensor coordinates (X) as part of Z
    % X is the 2x1 block top-right
    X = Z(1:2, 3:end);
    % Y is the nxn block bottom-right
    Y = Z(3:end, 3:end);
    
    % Objective: Minimize the error in distance constraints
    minimize(0) % Feasibility problem, or minimize slack variables for noise
    
    subject to
        % The Relaxation: Z must be Positive Semidefinite
        Z >= 0;
        
        % The Identity constraint: Top-left 2x2 must be Identity matrix
        Z(1:2, 1:2) == eye(2);
        
        % Distance constraints to Anchors
        % ||x_i - a_k||^2 = a_k'*a_k - 2*a_k'*x_i + Y_ii
        for k = 1:n_anchors
            anchors(:,k)'*anchors(:,k) - 2*anchors(:,k)'*X + Y(1,1) == d_sq(k);
        end
cvx_end

% 3. Results
fprintf('True Position: (%.2f, %.2f)\n', true_sensor(1), true_sensor(2));
fprintf('SDR Position:  (%.2f, %.2f)\n', X(1), X(2));

% Check if relaxation was 'tight' (is rank 2?)
eigs_Z = eig(Z);
fprintf('Eigenvalues of Z: %.4f, %.4f, %.4f\n', eigs_Z(end), eigs_Z(end-1), eigs_Z(end-2));