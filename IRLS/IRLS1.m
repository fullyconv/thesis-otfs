clc
clear all
close all

% -------- Veri hazırlığı (User Provided) --------
n_objects = 10;
pis = 100*randn(n_objects,2);          % 10 ölçüm noktası (Sensörler/Yansıtıcılar)
xs_true = [20,50];                     % Bilinmeyen Hedef Nokta (Estimate edeceğiz)
ys = [150,200];                        % Bilinen Sabit Nokta (Referans/Tx)

% Gerçek mesafeler
d_yp = sqrt(sum((ys - pis).^2,2));     % ys -> pis (Bilinen yol)
d_xp = sqrt(sum((xs_true - pis).^2,2));% xs -> pis (Bilinmeyen yol)
baseline_true = norm(xs_true - ys);    % xs -> ys (Bilinmeyen baseline)

% Ölçüm (Model: Toplam yol - Doğrudan yol + Gürültü)
% Model: h(x) = |y-p| + |x-p| - |x-y|
d_meas = d_yp + d_xp - baseline_true + exprnd(2,n_objects,1); 

% -------- Çözüm: Gauss-Newton (Iterative Least Squares) --------

% 1. Başlangıç Tahmini (Initial Guess)
% Rastgele bir noktadan veya ağırlık merkezinden başlayabiliriz
x_est = [0, 0]; 

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

% -------- Görselleştirme --------
figure('Color', 'w');
hold on; grid on; axis equal;

% 1. Sabitler
plot(pis(:,1), pis(:,2), 'ks', 'MarkerSize', 8, 'DisplayName', 'Sensors (pis)');
plot(ys(1), ys(2), 'bd', 'MarkerSize', 10, 'MarkerFaceColor', 'b', 'DisplayName', 'Anchor (ys)');

% 2. Gerçek Hedef
plot(xs_true(1), xs_true(2), 'gp', 'MarkerSize', 12, 'MarkerFaceColor', 'g', 'DisplayName', 'True Target (xs)');

% 3. Tahmin Yolu
plot(path_history(:,1), path_history(:,2), 'r.-', 'LineWidth', 1.5, 'DisplayName', 'Optimization Path');
plot(x_est(1), x_est(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Final Est');

legend('Location', 'best');
title(['Localization Result (RMSE: ' num2str(rmse, '%.4f') ')']);
xlabel('X Position'); ylabel('Y Position');

% Ellipses çizimi (Opsiyonel - Görsel kontrol için)
% İlk 3 ölçüm için elipsleri çiz
t = linspace(0, 2*pi, 100);
for i = 1:min(3, n_objects)
    % Bu bir bistatic range denklemidir, görselleştirme karmaşıktır.
    % Basitlik için sadece sensörleri hedefe bağlayan çizgileri çiziyoruz.
    line([pis(i,1) x_est(1)], [pis(i,2) x_est(2)], 'Color', [0.8 0.8 0.8], 'LineStyle', ':');
end