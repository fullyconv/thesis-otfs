clc; clear; close all

n_objects = 100;
pis = 100*randn(n_objects,2);
xs  = [50,30];
ys  = [150,200];

d_yp = sqrt(sum((ys - pis).^2,2));
d_xp = sqrt(sum((xs - pis).^2,2));

d_meas = d_yp + d_xp + 2*randn(n_objects,1) - norm(xs - ys);

% init
x_est  = [0,0];
xy_est = norm(x_est - ys);

for k = 1:10

    d_xp_pred = sqrt(sum((pis - x_est).^2,2));          % n x 1

    R = d_meas - (d_yp.' + d_xp_pred.' - xy_est);      % n x n (implicit expansion)
    C = R.^2;                                          % cost matrix (constant now)

    cvx_begin quiet
        variable P(n_objects,n_objects)
        minimize( sum(sum(C .* P)) )   % <-- LINEAR in P
        subject to
            P >= 0
            sum(P,1) == 1
            sum(P,2) == 1
    cvx_end

    [~, idx] = max(P, [], 2);
    Pperm = zeros(n_objects);
    for i=1:n_objects
        Pperm(i, idx(i)) = 1;
    end

    % -------- (2) x UPDATE: SOCP with fixed Pperm --------
    pis2  = Pperm * pis;
    dyp2  = d_yp;
    ci    = d_meas - dyp2 + xy_est;    % target for z ≈ d_xp

    cvx_begin quiet
        variable x_est(1,2)
        variable z(n_objects)
        minimize( sum_square(z - ci) )
        subject to
            z >= 0
            for i = 1:n_objects
                norm(x_est - pis2(i,:)) <= z(i);
            end
    cvx_end

    xy_est = norm(x_est - ys);
    fprintf("step %d \t x_est = %f %f \t xy_est=%f\n", k, x_est(1), x_est(2), xy_est)
end

fprintf('Tahmin edilen konum: (%.4f , %.4f)\n', x_est(1), x_est(2));
fprintf('Gerçek konum        : (%.4f , %.4f)\n', xs(1), xs(2));

figure; hold on; grid on;
plot(pis(:,1), pis(:,2), 'ko', 'MarkerSize', 6, 'DisplayName', 'pis');
plot(ys(1), ys(2), 'bs', 'MarkerSize', 10, 'DisplayName', 'B (ys)');
plot(xs(1), xs(2), 'r^', 'MarkerSize', 10, 'DisplayName', 'A (gerçek)');
plot(x_est(1), x_est(2), 'gd', 'MarkerSize', 12, 'DisplayName', 'A (tahmin)');
legend('Location', 'best');
xlabel('X'); ylabel('Y'); title('Konum tahmini (Alternating CVX)');
