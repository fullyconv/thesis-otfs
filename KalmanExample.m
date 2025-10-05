rx=[10,0];
tx=[-10,0];
tau=40;

%%
f1f2=vecnorm(rx-tx)/2;

cx = (rx(1)+tx(1))/2; 
cy = (rx(2)+tx(2))/2;     % center

a = tau/2; b = sqrt(a^2-f1f2^2);       % semi-axes
theta = deg2rad(0);% rotation angle in radians


%%
figure; hold on;
plot_ellipse(cx, cy, a, b, theta, 'g-', 'LineWidth',1.5);

% Plotting the positions

plot(rx(1), rx(2), 'ro', 'DisplayName', 'Receiver (rx)'); % Receiver positions in red
plot(tx(1), tx(2), 'bo', 'DisplayName', 'Transmitter (tx)'); % Transmitter positions in blue

% Adding labels and legend
xlabel('X Position');
ylabel('Y Position');
title('Positions of Receiver and Transmitter');
set(gca, 'Color', 'w'); % Set the background color to white
grid on; % Turn on the grid
set(gca, 'GridColor', 'k'); % Set grid color to black for better visibility
hold off

[phat] = KalmanCalc(rx,tx, tau,40)
