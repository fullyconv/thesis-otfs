function a = steering_vector_2d(positions, lambda, phi)
%STEERING_VECTOR_2D Steering vector for an arbitrary 2-D array geometry
%   a = steering_vector_2d(positions, lambda, phi)
%   INPUTS
%       positions : N-by-2 matrix, each row = [x_i y_i] (in metres)
%       lambda    : wavelength (metres)
%       phi       : azimuth angle of the look direction, in degrees.
%                   Angle is measured CCW from the +x-axis (broadside = 0°).
%   OUTPUT
%       a         : N-by-1 complex steering vector (unit-norm)
%   The formula used is
%       a_i = exp( -j * 2*pi/lambda * ( x_i*cos(phi) + y_i*sin(phi) ) )
%   This function is geometry-agnostic - you can pass the coordinates of a
%   uniform rectangular array, a circular array, a random sparse layout, etc.
%   Example:
%       % 8-element uniform rectangular array (URA) - 2x4 grid,
%       % half-wavelength spacing, wavelength = 0.03 m (10 GHz)
%       lambda = 0.03;
%       d      = lambda/2;
%       [X,Y]  = meshgrid(0:3,0:1);   % 4 columns (x), 2 rows (y)
%       pos    = [X(:)*d , Y(:)*d];   % Nx2 list of (x,y) coordinates
%       phi    = 45;                  % steer 45° from +x-axis
%       a      = steering_vector_2d(pos,lambda,phi);
%
% 1. Convert angle to radians (MATLAB trig works in rad)
phi_rad = deg2rad(phi);
% 2. Compute the projection of each element position onto the wave-vector
%    direction (k = [cosφ sinφ]ᵀ). This gives the electric-path length
%    difference in metres for each element.
proj = positions * [cos(phi_rad); sin(phi_rad)];  % Nx1 vector
% 3. Phase term (2π/λ) * projection --- the minus sign follows the
%    "receive" convention (plane wave arriving from φ).
a = exp(-1i * proj * (2*pi./lambda) );
%
end