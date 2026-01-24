clc; clear; close all

n_objects = 10;
pis = 200*randn(n_objects,2);
xs  = [50,30];
ys  = [150,200];

d_yp = sqrt(sum((ys - pis).^2,2));
d_xp = sqrt(sum((xs - pis).^2,2));

perm_true = eye(n_objects);
perm_true = perm_true(randperm(n_objects),:);

d_meas = perm_true*d_yp + perm_true*d_xp + 2*randn(n_objects,1) - norm(xs - ys);

% init
x_est  = [50,30];
xy_est = norm(x_est - ys);

P=eye(n_objects);
for k=1:15
    cvx_begin quiet
        variable x_est(1,2)
        variable z(n_objects)
        ci = d_meas - P*d_yp+xy_est; 
        minimize( sum_square( z - ci ) )    
        subject to
            pis2=P*pis;
            for i = 1:n_objects
                norm(x_est - pis2(i,:)) <= z(i);
            end
    cvx_end
    xy_est=norm(x_est - ys);
    fprintf("step %d \t xest=%f %f\n",k,x_est(1),x_est(2))
end

%% -------- Sonuç --------
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