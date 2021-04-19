clc; clear; close all;
%% Parameters
load('Nets/net1.mat')
userMass=80; %kg
sleepTime = 0.01; % sec



%% Initialize yarp Network
yarp.Network.init()

human_kin_dyn_port = yarp.BufferedPortBottle();
human_kin_dyn_port.open('/humanKinDyn:i');
yarp.Network.connect('/humanDataAcquisition/humanKinDyn:o', '/humanKinDyn:i');

lifting_started_port = yarp.BufferedPortBottle();
lifting_started_port.open('/liftingStarted:i');
yarp.Network.connect('/liftingStarted:o', '/liftingStarted:i');

lifting_ended_port = yarp.BufferedPortBottle();
lifting_ended_port.open('/liftingEnded:i');
is_connected=yarp.Network.connect('/liftingEnded:o', '/liftingEnded:i');

risk_port = yarp.BufferedPortVector();
risk_port.open('/estimatedRisk:o');


human_kin_dynBottle= yarp.Bottle;
liftingStartedBottle= yarp.Bottle;
liftingEndedBottle= yarp.Bottle;
liftingStarted = true;
liftingEnded = false;



while true
    human_kin_dynBottle = human_kin_dyn_port.read(false);
    liftingStartedBottle = lifting_started_port.read(false);
    liftingEndedBottle = lifting_ended_port.read(false);
    
    %     if isequal(liftingStartedBottle,[])
    if sum(size(liftingStartedBottle))==0
        liftingStarted = false;
    else
        disp(liftingStartedBottle.toString());
        liftingStarted = liftingStartedBottle.get(0).asBool();
        if liftingStarted
            disp('lifting is started: collecting data ...' )
        end
    end
    
    
    %     if isequal(liftingEndedBottle, [])
    if sum(size(liftingEndedBottle))==0
        liftingEnded = false;
    else
        disp(liftingEndedBottle.toString());
        liftingEnded = liftingEndedBottle.get(0).asBool();
        if liftingEnded
            disp('lifting is ended: estimaing the risk ...' )
        end
    end
    
    if liftingStarted && ~liftingEnded
        %         if isequal(human_kin_dyn, [])
        if sum(size(human_kin_dynBottle))==0
            disp('variable does  not exist');
        else
            kinDynInfo= human_kin_dynBottle.toString();
            disp(kinDynInfo);
        end
    end
    
    pause(sleepTime)
end


risk_port.close();
yarp.Network.fini();
