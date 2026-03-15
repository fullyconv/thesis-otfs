function [a] = createSteeringVector(ArrayGeometry,lambda,phi,d,N_ant)

switch  ArrayGeometry
    case 'ULA'
        % d = varargin{1};     % Antenler arası mesafe (Genelde lambda_orta / 2)
        % N_ant = varargin{2}; % Anten sayısı
        
        % 1. Anten indekslerini oluştur (Sütun vektörü: N_ant x 1)
        % 0, 1, 2, ..., N-1
        n = (0:N_ant-1)';
        
        % 2. Dalga numarasını hesapla (Satır vektörü: 1 x M)
        % k = 2*pi / lambda
        k = (2 * pi ./ lambda)'; 
        
        % 3. ULA Faz Denklemi: exp(j * n * k * d * sin(phi))
        % phi = 0 olduğunda (broadside) tüm antenler aynı fazdadır.
        % n*d*sin(phi) (N_ant x 1) ile k (1 x M) çarpılarak (N_ant x M) matris elde edilir.
        a = exp(1j * (n * d * sin(phi)) * k);
    % case 'Rectangular'
    %     dx=varargin{1};
    %     Nx=varargin{2};
    %     dy=varargin{3};
    %     Ny=varargin{4};
    %     x_vec=(0:Nx-1)*dx;
    %     y_vec=(0:Ny-1)*dy;
    %     [X, Y] = meshgrid(x_vec, y_vec);  % MATLAB’s default: X varies along columns,
    %     positions = [X(:) , Y(:)];              % (Nx*Ny)‑by‑2 matrix [x y]
    %     a = steering_vector_2d(positions, lambda, phi);
    case 'Circular'
            R = d; % Yarıçap

            % 1. Antenlerin dizilim açılarını oluştur (psi_n)
            % Resimdeki p_n'lerin açısal konumu
            psi_n = linspace(0, 2*pi, N_ant + 1)';
            psi_n = psi_n(1:end-1); % (N_ant x 1)

            % 2. Dalga numarası k = 2*pi / lambda
            % lambda (M x 1) olduğu için k da (M x 1) olur
            k = 2 * pi ./ lambda; 

            % 3. Resimdeki formülü uygula: exp(j * k * R * cos(theta - psi_n))
            % theta = phi (gelen sinyal açısı)
            % Boyutlandırma için 'broadcasting' kullanıyoruz:
            % (N_ant x 1) ve (1 x M) matris çarpımı/işlemi

            angle_diff = cos(phi - psi_n); % (N_ant x 1)
            % 360*angle_diff'/(2*pi)
            % a matrisi: (N_ant x M)
            % Her satır bir anteni, her sütun bir frekansı temsil eder
            a = exp(1j * (k') .* R .* angle_diff);
    otherwise
        error('Invalid Inputs\n')
end



end

