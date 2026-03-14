clc
clear all
close all
% -------- Veri hazırlığı --------
n_objects = 10;
pis = 100*randn(n_objects,2);          % 20 ölçüm noktası (20x2)
xs = [50,50];                            % Sabit nokta A
ys = [150,200];                        % Sabit nokta B

% Gerçek mesafeler
d_yp = sqrt(sum((ys - pis).^2,2));     % B -> pis
d_xp = sqrt(sum((xs - pis).^2,2));     % A -> pis

% Ölçüm (gerçek + gürültü)
d_meas = d_yp + d_xp + 2*randn(n_objects,1)-norm(xs - ys); % 20x1
xy_est=0;
estimates=[];
for k=1:10
    cvx_begin quiet
        variable x_est(1,2)
        variable z(n_objects)
        ci = d_meas - d_yp+xy_est; 
        minimize( sum_square( z - ci ) )    
        subject to
            for i = 1:n_objects
                norm(x_est - pis(i,:)) <= z(i);
            end
    cvx_end
    xy_est=norm(x_est - ys);
    fprintf("step %d \t xest=%f %f\n",k,x_est(1),x_est(2))
    estimates=[estimates;x_est];
end

% -------- Sonuç --------
fprintf('Tahmin edilen konum: (%.4f , %.4f)\n', x_est(1), x_est(2));
fprintf('Gerçek konum        : (%.4f , %.4f)\n', xs(1), xs(2));

figure(1)
plot(estimates(:,1),estimates(:,2),"Marker","*","Color","g");hold on
plot(xs(:,1),xs(:,2),"Marker","v","Color","r");hold off

% Görselleştirme (isteğe bağlı)
figure(2); hold on; grid on;
plot(pis(:,1), pis(:,2), 'ko', 'MarkerSize', 8, 'DisplayName', 'Ölçüm noktaları');
plot(ys(1), ys(2), 'bs', 'MarkerSize', 10, 'DisplayName', 'B (ys)');
plot(xs(1), xs(2), 'r^', 'MarkerSize', 10, 'DisplayName', 'A (gerçek)');
plot(x_est(1), x_est(2), 'gd', 'MarkerSize', 12, 'DisplayName', 'A (tahmin)');
legend('Location', 'best');
xlabel('X'); ylabel('Y'); title('Konum tahmini - (CVX)');