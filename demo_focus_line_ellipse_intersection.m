function demo_focus_line_ellipse_intersection()
    % --- Ellipse definition via foci and semi-major axis 'a'
    F1 = [0, 0];
    F2 = [8, 0];
    a  = 6;                          % semi-major (must satisfy 2a > |F1-F2|)
    assert(2*a > norm(F2-F1), 'Invalid ellipse: 2a must exceed focal distance.');

    % Pick a ray direction from F1 (e.g., 35 degrees)
    ang_deg = 35;
    u = [cosd(ang_deg), sind(ang_deg)];
    u = u / norm(u);

    % Compute intersection from F1 along +u (one point on the ellipse)
    [t_pos, p_pos, flag_pos] = focus_ray_ellipse_intersection(F1, F2, a, u);
    
    % (Optional) also get the other intersection by shooting the opposite ray
    [t_neg, p_neg, flag_neg] = focus_ray_ellipse_intersection(F1, F2, a, -u);

    % --- Plot the ellipse, foci, ray, and intersection(s)
    figure; hold on; axis equal; grid on;
    title('Line from F_1 intersecting the ellipse (focal definition)');
    xlabel('x'); ylabel('y');

    % Plot ellipse built from (F1,F2,a)
    plot_ellipse_from_foci(F1, F2, a, 'b-', 1.5);

    % Plot foci
    plot(F1(1), F1(2), 'ro', 'MarkerFaceColor','r'); 
    plot(F2(1), F2(2), 'ro', 'MarkerFaceColor','r'); 
    legend_entries = {'Ellipse','Foci'};

    % Plot the ray from F1
    L = 3*a; % length for drawing the ray
    ray_end = F1 + L*u;
    plot([F1(1), ray_end(1)], [F1(2), ray_end(2)], 'k--');
    legend_entries{end+1} = 'Ray from F_1';

    % Plot intersections if valid
    if flag_pos
        plot(p_pos(1), p_pos(2), 'gs', 'MarkerFaceColor','g', 'MarkerSize',8);
        legend_entries{end+1} = 'Intersection (+u)';
    end
    if flag_neg
        plot(p_neg(1), p_neg(2), 'ms', 'MarkerFaceColor','m', 'MarkerSize',8);
        legend_entries{end+1} = 'Intersection (-u)';
    end

    legend(legend_entries{:}, 'Location', 'best');
end

function [t, p, ok] = focus_ray_ellipse_intersection(F1, F2, a, u)
    % Intersection of the ray p(t)=F1 + t*u, t>=0 with ellipse |p-F1|+|p-F2|=2a
    % Closed-form (with numeric fallback if near-tangent)
    s = F1 - F2;
    num   = (2*a)^2 - dot(s,s);
    denom = 2*(2*a + dot(s,u));
    epsd  = 1e-10;

    if abs(denom) > epsd
        t = num / denom;
        if t >= 0 && (2*a - t) >= 0
            p = F1 + t*u;
            ok = true;
            return;
        end
    end

    % Fallback: numeric solve for t >= 0
    f = @(tt) norm(s + tt*u) + tt - 2*a; % |F1-F2 + t u| + t - 2a = 0
    % Try to bracket a root on [0, 5a] (expand if needed)
    tL = 0; tR = 5*a;
    fL = f(tL); fR = f(tR);
    expand = 0;
    while fL*fR > 0 && expand < 5
        tR = tR * 2; fR = f(tR); expand = expand + 1;
    end
    if fL*fR <= 0
        t = fzero(f, [tL, tR]);
        if t >= -1e-9 % tolerance
            t = max(t, 0);
            p = F1 + t*u;
            ok = true;
            return;
        end
    end
    % No valid intersection along this ray
    t = NaN; p = [NaN, NaN]; ok = false;
end

function plot_ellipse_from_foci(F1, F2, a, ls, lw)
    % Plot ellipse given foci F1,F2 and semi-major axis a
    C = 0.5*(F1 + F2);               % center
    c = 0.5*norm(F2 - F1);           % focal distance
    b = sqrt(a^2 - c^2);             % semi-minor
    theta = atan2(F2(2)-F1(2), F2(1)-F1(1)); % orientation of major axis

    t = linspace(0, 2*pi, 400);
    x_local = a*cos(t); 
    y_local = b*sin(t);
    R = [cos(theta), -sin(theta); sin(theta), cos(theta)];
    XY = R * [x_local; y_local];
    x = XY(1,:) + C(1);
    y = XY(2,:) + C(2);
    plot(x, y, ls, 'LineWidth', lw);
end
