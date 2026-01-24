clc; clear; close all

n_objects = 10;
pis = 100*randn(n_objects,2);
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


for k = 1

    cvx_begin quiet
        variable P(n_objects,n_objects)
        d_xp_pred = sqrt(sum((pis - x_est).^2,2));          % n x 1
    
        R = d_meas - (P*d_yp + (P*(d_xp_pred)) - xy_est);      % n x n (implicit expansion)
        
        minimize(sum_square(R) -5*sum(sum(entr(P+1e-8))))   % <-- LINEAR in P
        subject to
            P >= 0
            sum(P,1) == 1
            sum(P,2) == 1
    cvx_end
end

for i=1:10
    figure(i)
    plot(P(i,:));hold on
    plot(perm_true(i,:));hold off
end

