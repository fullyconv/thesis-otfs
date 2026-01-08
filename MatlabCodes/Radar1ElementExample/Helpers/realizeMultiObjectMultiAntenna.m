function [env] = realizeMultiObjectMultiAntenna(env,tx)

%% Single target path from your defaults (approx):


tau = 2*env.targetDistance/env.c0;       % seconds
lambda = env.c0/env.fc;
fd  = 2*env.targetVelocity/lambda;       % Hz (monostatic)
alpha = 1.0 * exp(1j*2*pi*rand);         % random phase

rx = env.realizeChannel(alpha, tau, fd, tx);


%% MULTI ANTENNA
f_list=(1:env.N)*env.deltaf;
Ytf = env.sense(rx,dd);

d=env.targetDistance;
theta=30; %degree

NAntenna=8;
phas_array=(0:NAntenna-1)'*2*pi*d*sind(theta)*f_list/physconst('LightSpeed');
Sensed_signal = Ytf .* permute(phas_array, [3 2 1]);

end