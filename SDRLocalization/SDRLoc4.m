% -------- Veri hazırlığı --------
n_objects = 20;
pis = 100*randn(n_objects,2);          % 20 ölçüm noktası (20x2)
xs = [90,10];                            % Sabit nokta A
ys = [150,200];                        % Sabit nokta B

% Gerçek mesafeler
d_yp = sqrt(sum((ys - pis).^2,2));     % B -> pis
d_xp = sqrt(sum((xs - pis).^2,2));     % A -> pis


% Ölçüm (gerçek + gürültü)
d_meas = d_yp + d_xp + 2*randn(n_objects,1); % 20x1

% -------- CVX Optimizasyonu --------
cvx_begin quiet
    variable x_est(1,2)                % Tahmini konum (x, y)
    variable z(n_objects)              % Yardımcı mesafe değişkeni (norm yerine)
    
    % Sabit terimi önceden hesaplayalım (c_i)
    ci = d_meas - d_yp; 
    
    % Amaç: sum( (z + ci).^2 ) minimize edilsin
    % CVX'te bir ifadenin karesini almak için 'square' veya 'sum_square' kullanılır.
    % Bu fonksiyonlar sadece içindeki ifade "doğrusal (affine)" ise çalışır.
    % z ve ci doğrusal olduğu için bu yazım DCP'ye uygundur.
    minimize( sum_square( z - ci ) )
    
    subject to
        % TEMEL TRICK: sqrt(...) ifadesini kısıt olarak yazıyoruz.
        % norm(A) <= B ifadesi CVX'te temel bir koni kısıtıdır ve convex'tir.
        for i = 1:n_objects
            norm(x_est - pis(i,:)) <= z(i);
        end
cvx_end

% -------- Sonuç --------
fprintf('Tahmin edilen konum: (%.4f , %.4f)\n', x_est(1), x_est(2));
fprintf('Gerçek konum        : (%.4f , %.4f)\n', xs(1), xs(2));

% Görselleştirme (isteğe bağlı)
figure; hold on; grid on;
plot(pis(:,1), pis(:,2), 'ko', 'MarkerSize', 8, 'DisplayName', 'Ölçüm noktaları');
plot(ys(1), ys(2), 'bs', 'MarkerSize', 10, 'DisplayName', 'B (ys)');
plot(xs(1), xs(2), 'r^', 'MarkerSize', 10, 'DisplayName', 'A (gerçek)');
plot(x_est(1), x_est(2), 'gd', 'MarkerSize', 12, 'DisplayName', 'A (tahmin)');
legend('Location', 'best');
xlabel('X'); ylabel('Y'); title('Konum tahmini - (CVX)');