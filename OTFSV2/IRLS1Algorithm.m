function [x_est,loss_history,path_history] = IRLS1Algorithm(pis,ys,d_meas)


x_est = [0, 0]; 
d_yp = sqrt(sum((ys - pis).^2,2));     % ys -> pis (Bilinen yol)
n_objects=size(pis,1);

max_iter = 20;
tol = 1e-6;
loss_history = [];
path_history = x_est;

fprintf('%5s | %10s | %10s | %10s\n', 'Iter', 'x(1)', 'x(2)', 'RMSE');
fprintf('--------------------------------------------------\n');

for k = 1:max_iter
    % --- A. Mevcut tahmine göre mesafeleri hesapla ---
    % Vektör: pis -> x_est
    vec_xp = x_est - pis;             % (Nx2)
    dist_xp = sqrt(sum(vec_xp.^2, 2));% (Nx1)
    
    % Vektör: ys -> x_est
    vec_xy = x_est - ys;              % (1x2)
    dist_xy = norm(vec_xy);           % Scalar
    
    % --- B. Hata (Residual) Hesapla ---
    % Model: h(x) = |y-p| + |x-p| - |x-y|
    % Not: d_yp sabittir, iterasyonda değişmez.
    h_val = d_yp + dist_xp - dist_xy; 
    
    residual = d_meas - h_val;        % (Meas - Model)
    
    % Loss takibi (RMSE)
    rmse = sqrt(mean(residual.^2));
    loss_history(end+1) = rmse;
    
    fprintf('%5d | %10.4f | %10.4f | %10.4f\n', k, x_est(1), x_est(2), rmse);
    
    if rmse < tol
        break;
    end
    
    % --- C. Jacobian Matrisi Oluştur (Linearization) ---
    % Türevler: d(|x-p|)/dx = (x-p) / |x-p| (Unit vector)
    % Türevler: d(-|x-y|)/dx = -(x-y) / |x-y|
    
    % Unit vectors from p_i to x_est
    u_px = vec_xp ./ dist_xp; 
    
    % Unit vector from y to x_est (repeated for N rows)
    u_yx = vec_xy / dist_xy;
    u_yx_rep = repmat(u_yx, n_objects, 1);
    
    % Jacobian (Nx2) -> J(i,:) = u_px(i) - u_yx
    H = u_px - u_yx_rep;
    
    % --- D. Güncelleme (Gauss-Newton Step) ---
    % Delta = (H' * H)^-1 * H' * residual
    delta = (H' * H) \ (H' * residual);
    
    x_est = x_est + delta';
    path_history = [path_history; x_est];
end


end