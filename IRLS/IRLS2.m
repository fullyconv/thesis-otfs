clc; clear all; close all;

% -------- Veri Hazırlığı --------
n_objects = 10;
pis = 100*randn(n_objects,2);          
xs_true = [50,50];                     
ys = [150,200];                        

% Gerçek Ölçümler
d_yp = sqrt(sum((ys - pis).^2,2));     
d_xp = sqrt(sum((xs_true - pis).^2,2));
baseline_true = norm(xs_true - ys);    

% Model: d_meas = |y-p| + |x-p| - |x-y| + noise
% Biz bunu şu formda çözeceğiz: |x-p| - |x-y| = C
d_meas = d_yp + d_xp - baseline_true + 2*randn(n_objects,1); 

% Sabit Terim (Constant)
% Denklemimiz: |x-p| - |x-y| = d_meas - |y-p|
rhs_constant = d_meas - d_yp; 

% -------- 2 Set Ağırlıklı (Explicit Geometric Weights) Çözüm --------
x_est = [0, 0]; % Başlangıç
max_iter = 20;
tol = 1e-6;
epsilon = 1e-6; % Bölme hatası önleyici

fprintf('%5s | %10s | %10s | %10s\n', 'Iter', 'x(1)', 'x(2)', 'RMSE');
fprintf('--------------------------------------------------\n');

for k = 1:max_iter
    % 1. Vektörleri Hesapla
    vec_xp = x_est - pis;              % (Nx2)
    vec_xy = x_est - ys;               % (1x2)
    
    % 2. Mesafeleri Hesapla (Euclidean Norm)
    dist_xp = sqrt(sum(vec_xp.^2, 2)); % (Nx1)
    dist_xy = norm(vec_xy);            % Scalar
    
    % ---------------------------------------------------------
    % Geometrik Ağırlıklar: Türevi doğrusallaştırmak için
    % d(|x-p|)/dx = (x-p) * (1/|x-p|)  -> (x-p) * w_xp
    % ---------------------------------------------------------
    
    w_xp = 1 ./ (dist_xp + epsilon);   % Set 1: x-p ağırlıkları (Nx1)
    w_xy = 1 / (dist_xy + epsilon);    % Set 2: x-y ağırlığı (Scalar)
    
    % ---------------------------------------------------------
        
    % (x-p) terimi için Jacobian satırları
    J_part1 = vec_xp .* w_xp; % Element-wise çarpım (Nx2)
    
    % (x-y) terimi için Jacobian (Tekrarlayan satırlar)
    J_part2 = repmat(vec_xy * w_xy, n_objects, 1);
    
    % Toplam Jacobian (Zincir kuralı: |x-p|'nin türevi - |x-y|'nin türevi)
    % Not: Formülümüz |x-p| - |x-y| olduğu için arada eksi var.
    H = J_part1 - J_part2; 
    
    % 4. Residual (Kalan Hata)
    % Model: |x-p| - |x-y|
    model_val = dist_xp - dist_xy;
    r = rhs_constant - model_val; % (Measured - Model)
    
    rmse = sqrt(mean(r.^2));
    fprintf('%5d | %10.4f | %10.4f | %10.4f\n', k, x_est(1), x_est(2), rmse);
    
    if rmse < tol
        break;
    end
    
    % 5. Newton Güncellemesi (Least Squares)
    % Delta = (H'H)^-1 * H'r
    delta = (H' * H) \ (H' * r);
    x_est = x_est + delta';
end

% --- Görselleştirme ---
figure('Color', 'w'); hold on; grid on; axis equal;
plot(pis(:,1), pis(:,2), 'ks', 'DisplayName', 'Sensors');
plot(ys(1), ys(2), 'bd', 'MarkerFaceColor','b', 'DisplayName', 'Anchor');
plot(xs_true(1), xs_true(2), 'gp', 'MarkerFaceColor','g', 'MarkerSize',12, 'DisplayName', 'True Target');
plot(x_est(1), x_est(2), 'rx', 'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', 'Result');
legend;
title('Geometric Weighting (Explicit Weights)');