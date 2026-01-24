function [a] = createSteeringVector(ArrayGeometry,lambda,phi,varargin)

switch ArrayGeometry
    case 'ULA'
        d=varargin{1};
        N=varargin{2};
        positions=[(0:N-1)*d;zeros(1,N)];
        a = steering_vector_2d(positions', lambda, phi);
    case 'Rectangular'
        dx=varargin{1};
        Nx=varargin{2};
        dy=varargin{3};
        Ny=varargin{4};
        x_vec=(0:Nx-1)*dx;
        y_vec=(0:Ny-1)*dy;
        [X, Y] = meshgrid(x_vec, y_vec); % MATLAB's default: X varies along columns,
        positions = [X(:) , Y(:)];       % (Nx*Ny)-by-2 matrix [x y]
        a = steering_vector_2d(positions, lambda, phi);
    case 'Circular'
        R=varargin{1};
        N=varargin{2};
        positions=[R*sin(linspace(0,2*pi,N+1))',R*cos(linspace(0,2*pi,N+1))'];
        positions=positions(1:end-1,:);
        a = steering_vector_2d(positions, lambda, phi);
    otherwise
        error('Invalid Inputs\n')
end

end