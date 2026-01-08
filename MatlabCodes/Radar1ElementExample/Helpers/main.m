f_list=(1:env.N)*env.deltaf;

rx=zeros(env.M*env.N,1);
Sensed_signal=zeros(env.M,env.N,env.NAntenna);

for target_idx=1:length(env.TargetList)
    % env.TargetList(target_idx)
    % env.TargetList(target_idx).Velocity
    targetDistance=vecnorm(env.TargetList(target_idx).Position);
    tau=2*targetDistance/env.c0;

    lambda = env.c0/env.fc;
    fd  = 2*env.targetVelocity/lambda;       % Hz (monostatic)
    alpha = 1.0 * exp(1j*2*pi*rand);         % random phase
    
    rxSignal = env.realizeChannel(alpha, tau, fd, tx);
    
    Ytf = WignerTransform(rxSignal, env.M, env.N);
    
    theta =atan2d(env.TargetList(2).Position(1), ...
    env.TargetList(2).Position(2));

    phas_array=(1:env.NAntenna)'*2*pi*env.antennaSpacing*sind(theta)*f_list/physconst('LightSpeed');
    phas_array=exp(1j*phas_array);
    Sensed_signal = Sensed_signal+(Ytf .* permute(phas_array, [3 2 1]));
    
end



%% MULTI ANTENNA
% Ytf = env.sense(rx,dd);
% 
% d=env.targetDistance;
% theta=30; %degree
% 
% NAntenna=8;
% phas_array=(0:NAntenna-1)'*2*pi*d*sind(theta)*f_list/physconst('LightSpeed');
% Sensed_signal = Ytf .* permute(phas_array, [3 2 1]);
