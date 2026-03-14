% env = Environment;
% env.SNRdB=100;
% env.N=32;
% env.targetDistance=30;
% env.targetVelocity=80;
% 
% bits = env.createDataBits();
% [tx,dd]   = env.createTxSignal(bits);
% % Single target path from your defaults (approx):
% tau = 2*env.targetDistance/env.c0;       % seconds
% lambda = env.c0/env.fc;
% fd  = 2*env.targetVelocity/lambda;       % Hz (monostatic)
% alpha = 1.0 * exp(1j*2*pi*rand);         % random phase
% 
% rx = env.realizeChannel(alpha, tau, fd, tx);
% 
% [estimatedVelocity,estimatedRange]= env.estimateRangeVelocity(rx,dd)
% 



%% MULTI OBJECT
env = Environment;
env.N=32;
env.M=64;

env.SNRdB=60;
bits = env.createDataBits();
bits(2:end,:)=0;bits(1,:)=[0,1];
[tx,dd]   = env.createTxSignal(bits);

n_objects=1;
velocityList=linspace(0,90,n_objects);
positionList=100*[randn(1,n_objects);randn(1,n_objects)];
rangeList=sqrt(positionList(1,:).^2+positionList(2,:).^2);

ObjectList=cell(1,length(positionList));
for i=1:size(positionList,2)
    ObjectList{i} = Platform(0, velocityList(i), positionList(:,i));
end
ObjectList=[ObjectList{:}];
env.TargetList=ObjectList;


%%

tau = 2*rangeList/env.c0;       % seconds
lambda = env.c0/env.fc;
fd  = 2*velocityList/lambda;       % Hz (monostatic)
alpha = 1.0 * exp(1j*2*pi*rand(1,n_objects));         % random phase
rx = env.realizeChannel(alpha, tau, fd, tx);

% Sensing receiver
Ytf = WignerTransform(rx, env.M, env.N);
Ydd = SFFT(Ytf, env.M, env.N);
imshow(Ydd)
% [estimatedVelocity,estimatedRange]= env.estimateRangeVelocity(rx,dd)
