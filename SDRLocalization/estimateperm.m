function [P] = estimateperm(pis, x_est,d_meas,d_yp ,xy_est)
n_objects=size(pis,1);
cvx_begin quiet
    variable P(n_objects,n_objects)
    d_xp_pred = sqrt(sum((pis - x_est).^2,2));          % n x 1

    R = d_meas - (P*d_yp + (P*(d_xp_pred)) - xy_est);      % n x n (implicit expansion)
    
    minimize(sum_square(R) -2*sum(sum(entr(P+1e-8))))   % <-- LINEAR in P
    subject to
        P >= 0
        sum(P,1) == 1
        sum(P,2) == 1
cvx_end

end