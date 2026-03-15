function [x_est, rmse_history] = IRLS2(d_meas, pis, ys, init_guess, max_iter, tol)
% IRLS2ALGORITHM Kestirim algoritmasını 2. Set Ağırlıklandırılmış formda çözer.
% Gauss-Newton ve Explicit Geometric Weights (Epsilon korumalı) kullanır.
%
% Girdiler:
%   d_meas     : Ölçülen toplam (bistatic) yollar |y-p| + |x-p| - |x-y| (Nx1 Vektörü)
%   pis        : Yansıtıcı (Sensor/RIS) konumları matrisi (Nx2 Matris)
%   ys         : Sabit istasyon veya Verici İHA (y) konumu (1x2 Vektör)
%   init_guess : (Opsiyonel) Başlangıç tahmin noktası (1x2 Vektör, Varsayılan: [0, 0])
%   max_iter   : (Opsiyonel) Maksimum iterasyon sayısı (Varsayılan: 20)
%   tol        : (Opsiyonel) Çıkış toleransı / Hata sınırı (Varsayılan: 1e-6)
%
% Çıktılar:
%   x_est        : Tahmin edilen hedef konumu (1x2 Vektör)
%   rmse_history : İterasyonlar boyunca RMSE hata geçmişi dizisi

    % Opsiyonel argümanların kontrolü
    if nargin < 4 || isempty(init_guess), init_guess = [0, 0]; end
    if nargin < 5 || isempty(max_iter), max_iter = 20; end
    if nargin < 6 || isempty(tol), tol = 1e-6; end

    n_objects = size(pis, 1);
    
    % Sabit olan d_yp (y ile p arası mesafeler) hesabı
    d_yp = sqrt(sum((ys - pis).^2, 2));     
    
    % Sabit Terim (Constant)
    % Denklemimiz: |x-p| - |x-y| = d_meas - |y-p|
    rhs_constant = d_meas - d_yp; 

    % -------- Çözüm Döngüsü --------
    x_est = init_guess;
    epsilon = 1e-6; % Bölme (Sıfır ile) hatası önleyici
    rmse_history = [];
    
    for k = 1:max_iter
        % 1. Vektörleri Hesapla
        vec_xp = x_est - pis;              % (Nx2)
        vec_xy = x_est - ys;               % (1x2)
        
        % 2. Mesafeleri Hesapla (Euclidean Norm)
        dist_xp = sqrt(sum(vec_xp.^2, 2)); % (Nx1)
        dist_xy = norm(vec_xy);            % Scalar
        
        % ---------------------------------------------------------
        % Geometrik Ağırlıklar: Türevi doğrusallaştırmak için
        % ---------------------------------------------------------
        w_xp = 1 ./ (dist_xp + epsilon);   % Set 1: x-p ağırlıkları (Nx1)
        w_xy = 1 / (dist_xy + epsilon);    % Set 2: x-y ağırlığı (Scalar)
        
        % (x-p) terimi için Jacobian satırları
        J_part1 = vec_xp .* w_xp;          % Element-wise çarpım (Nx2)
        
        % (x-y) terimi için Jacobian (Tekrarlayan satırlar)
        J_part2 = repmat(vec_xy * w_xy, n_objects, 1);
        
        % Toplam Jacobian matrisi
        H = J_part1 - J_part2; 
        
        % 4. Residual (Kalan Hata)
        % Model: |x-p| - |x-y|
        model_val = dist_xp - dist_xy;
        r = rhs_constant - model_val;      % (Measured - Model)
        
        rmse = sqrt(mean(r.^2));
        rmse_history(end+1) = rmse;        % Hata geçmişine kaydet
        
        if rmse < tol
            break;
        end
        
        % 5. Newton Güncellemesi (Least Squares Formülü)
        % Delta = (H'H)^-1 * H'r
        delta = (H' * H) \ (H' * r);
        x_est = x_est + delta';
    end
end