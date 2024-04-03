function [x,y,theta,internalStateOut] = estRun(time, dt, internalStateIn, steeringAngle, pedalSpeed, measurement)
% In this function you implement your estimator. The function arguments
% are:
%  time: current time in [s]
%  dt: current time step [s]
%  internalStateIn: the estimator internal state, definition up to you.
%  steeringAngle: the steering angle of the bike, gamma, [rad]
%  pedalSpeed: the rotational speed of the pedal, omega, [rad/s]
%  measurement: the position measurement valid at the current time step
%
% Note: the measurement is a 2D vector, of x-y position measurement.
%  The measurement sensor may fail to return data, in which case the
%  measurement is given as NaN (not a number).
%
% The function has four outputs:
%  x: your current best estimate for the bicycle's x-position
%  y: your current best estimate for the bicycle's y-position
%  theta: your current best estimate for the bicycle's rotation theta
%  internalState: the estimator's internal state, in a format that can be understood by the next call to this function

% Example code only, you'll want to heavily modify this.
% this needs to correspond to your estInitialize function:

x = internalStateIn.x;
y = internalStateIn.y;
theta = internalStateIn.theta;
color = internalStateIn.color;

x = x + pedalSpeed;
y = y + pedalSpeed;

if ~isnan(measurement(1)) & ~isnan(measurement(2))
    % have a valid measurement
    x = measurement(1);
    y = measurement(2);
    theta = theta + 1;
end

% we're unreliable about our favourite colour: 
if strcmp(color, 'green')
    color = 'red';
else
    color = 'green';
end

%% OUTPUTS %%
% Update the internal state (will be passed as an argument to the function
% at next run), must obviously be compatible with the format of
% internalStateIn:

internalStateOut.x = x;
internalStateOut.y = y;
internalStateOut.theta = theta;
internalStateOut.color = color;

end