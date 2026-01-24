%%
N = 10;      % Elements in array
d = 0.2;     % sensor spacing half wavelength
elementPos = (0:N-1)*d;
sv = steervec(elementPos,[30;0]);

%% ULA
lambda = 0.03;
d      = lambda/2;
[a] = createSteeringVector('ULA',lambda,30,d,10);

%% Rectangular
phi=[0,30];
lambda = [0.03];
dx     = lambda(1)/2;
Nx     = 30;
dy     = lambda(1)/2;
Ny     = 20;
[a] = createSteeringVector('Rectangular',lambda,phi,dx,Nx,dy,Ny);

%% Circular
phi=[0];
lambda = [0.03,0.04];
N      = 30;
R      = lambda(1);
[a] = createSteeringVector('Circular',lambda,phi,R,N);

%%
M=64;
N=16;
Ytf=randn(M,N);
f_start=500e9;
deltaf=10e3;
phi=[0];
f=f_start+deltaf*(0:M-1);
lambda = 3e8*1./f;
N      = 30;
R      = lambda(1);
[a] = createSteeringVector('Circular',lambda,phi,R,N);