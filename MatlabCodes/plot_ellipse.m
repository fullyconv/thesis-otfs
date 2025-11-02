function plot_ellipse(cx, cy, a, b, theta, varargin)
%PLOT_ELLIPSE  Plot an ellipse with given center, axes, and angle.
%   plot_ellipse(cx, cy, a, b, theta)
%   - (cx, cy): center coordinates
%   - a: semi-major axis length
%   - b: semi-minor axis length
%   - theta: rotation angle (radians)
%   Optional plot args (e.g. 'r--', 'LineWidth',2) can be passed.

    % Parametric angle
    t = linspace(0, 2*pi, 200);

    % Ellipse in local (unrotated) frame
    x_local = a * cos(t);
    y_local = b * sin(t);

    % Rotation matrix
    R = [cos(theta) -sin(theta);
         sin(theta)  cos(theta)];

    % Rotate and shift
    pts = R * [x_local; y_local];
    x = pts(1,:) + cx;
    y = pts(2,:) + cy;

    % Plot
    plot(x, y, varargin{:}); hold on;
    axis equal;
    xlabel('X'); ylabel('Y');
    title('Ellipse with center, major/minor axes, and angle');

    % Draw major/minor axes
    line([cx, cx + a*cos(theta)], [cy, cy + a*sin(theta)], ...
         'Color', 'k', 'LineStyle', '--');
    line([cx, cx - b*sin(theta)], [cy, cy + b*cos(theta)], ...
         'Color', 'k', 'LineStyle', '--');

    % Mark center
    plot(cx, cy, 'ko', 'MarkerFaceColor', 'k');
end
