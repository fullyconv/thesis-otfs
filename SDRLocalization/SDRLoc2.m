% Tightened SDR for: ||x-p|| + ||x|| = C
clear; clc;

% 1. Setup
p = [0, 0; 10, 0; 5, 8]'; 
y_known = [2, 2]';        
true_x = [4, 3]';         
n_anchors = size(p, 2);

% Generate d_j
d = zeros(n_anchors, 1);
for j = 1:n_anchors
    d(j) = norm(true_x - p(:,j)) + norm(y_known - p(:,j)) + norm(true_x);
end

% 2. Tightened SDR Optimization
cvx_begin sdp
    variable Z(3, 3) symmetric
    variable u(n_anchors) 
    variable v
    
    x_est = Z(1:2, 3);
    Y = Z(3, 3); % Represents ||x||^2
    
    % OBJECTIVE: Minimize error + maximize tightness
    % We minimize Y and u to "shrink" the relaxation onto the boundary
    minimize( sum(u) + v + 10*trace(Z) ) 
    
    subject to
        Z >= 0;
        Z(1:2, 1:2) == eye(2);
        
        for j = 1:n_anchors
            const_j = norm(y_known - p(:,j));
            rhs = d(j) - const_j;
            
            % The core distance equation (Linear)
            u(j) + v == rhs;
            
            % Lifted distance: ||x - p_j||^2 <= u_j^2
            % Using Schur Complement for u_j^2 >= ||x-p_j||^2
            [u(j), (x_est - p(1:2,j))'; (x_est - p(1:2,j)), u(j)*eye(2)] >= 0;
            
            u(j) >= 0;
        end
        
        % Tighten v: v^2 >= ||x||^2
        [v, x_est'; x_est, v*eye(2)] >= 0;
        v >= 0;
cvx_end

% 3. Check Rank (Is the relaxation tight?)
e = eig(Z);
rank_ratio = e(end-1)/e(end); % Should be very small for a Rank-1 solution

fprintf('--- Results ---\n');
fprintf('True x: [%.2f, %.2f]\n', true_x(1), true_x(2));
fprintf('SDR x:  [%.2f, %.2f]\n', x_est(1), x_est(2));
fprintf('Rank Ratio (smaller is better): %.4f\n', rank_ratio);