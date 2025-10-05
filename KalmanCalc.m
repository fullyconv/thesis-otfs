function [phat] = KalmanCalc(rx,tx, tau,theta)


% mean_position=(rx+tx)/2;
% rx=rx-mean_position;
% tx=tx-mean_position;
% 
% theta_rx=atan2d(rx(1),rx(2));
% theta_tx=atan2d(tx(1),tx(2));
% 
% r_rx=vecnorm(rx);
% r_tx=vecnorm(tx);


f1f2=vecnorm(rx-tx)/2;
a = tau/2; 
b = sqrt(a^2-f1f2^2);       % semi-axes
m=tand(theta);
xhat=sqrt((a^2*b^2)/(b^2+a^2*-m^2));
yhat=tand(theta)*xhat;
phat=[xhat,yhat];

end