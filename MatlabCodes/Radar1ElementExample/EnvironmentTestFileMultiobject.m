env = Environment;
env.SNRdB=100;
env.N=32;
env.targetDistance=30;
env.targetVelocity=80;

bits = env.createDataBits();
[tx,dd]   = env.createTxSignal(bits);
% Single target path from your defaults (approx):
tau = 2*env.targetDistance/env.c0;       % seconds
lambda = env.c0/env.fc;
fd  = 2*env.targetVelocity/lambda;       % Hz (monostatic)
alpha = 1.0 * exp(1j*2*pi*rand);         % random phase

rx = env.realizeChannel(alpha, tau, fd, tx);

[estimatedVelocity,estimatedRange]= env.estimateRangeVelocity(rx,dd)




%% MULTI OBJECT
% 
velocityList=linspace(0,90,10);
positionList=100*[randn(1,10);randn(1,10)];
rangeList=sqrt(positionList(1,:).^2+positionList(2,:).^2);

ObjectList=cell(1,length(positionList));
for i=1:length(positionList)
    ObjectList{i} = Platform(0, velocityList(i), positionList(:,i));
end
ObjectList=[ObjectList{:}];
env.TargetList=ObjectList;


%%

tau = 2*rangeList/env.c0;       % seconds
lambda = env.c0/env.fc;
fd  = 2*velocityList/lambda;       % Hz (monostatic)
alpha = 1.0 * exp(1j*2*pi*rand(1,10));         % random phase
rx = env.realizeChannel(alpha, tau, fd, tx);
[estimatedVelocity,estimatedRange]= env.estimateRangeVelocity(rx,dd)
