clc; clear; close all;
%% Parameters
load('Nets/net1.mat')
userMass=80; %kg
userWeight=userMass *9.81;
sleepTime = 0.01; % sec

NoFeatures= 60;
NoFinalSamples = 50;

humanKinDynTimeSeries=[]; % all the time series data
humanKinDynVector=zeros(1,NoFeatures); % only the last vector
scalingVector=ones(1,NoFeatures);
scalingVector(1,end-11:end)=1/userWeight; % end-11:end ==> related to the external wrenches

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
liftingStarted = false;
liftingEnded = false;


while true
    
    NoInfo = lifting_started_port.getPendingReads();
    if NoInfo >0
        liftingStartedBottle = lifting_started_port.read(false);
        liftingStartedInfo= liftingStartedBottle.toString();
        disp(liftingStartedInfo)
        
        liftingStarted = liftingStartedBottle.get(0).asBool();
        if liftingStarted
            disp('lifting is started: collecting data ...' )
        end
    end
    
    NoInfo = lifting_ended_port.getPendingReads();
    if NoInfo >0
        liftingEndedBottle = lifting_ended_port.read(false);
        liftingEndedInfo= liftingEndedBottle.toString();
        disp(liftingEndedInfo)
        
        liftingEnded = liftingEndedBottle.get(0).asBool();
        if liftingEnded
            disp('lifting is ended: estimaing the risk ...' )
            break;
        end
    end
    
    %     kinDynInfo= human_kin_dynBottle.toString();
    %     disp(kinDynInfo)
    
    
    
    if liftingStarted && ~liftingEnded
        NoInfo = human_kin_dyn_port.getPendingReads();
        if NoInfo>0
            human_kin_dynBottle = human_kin_dyn_port.read(false);
            
            kinDynInfo= human_kin_dynBottle.toString();
            disp(kinDynInfo);
            
            % get the information from the yarp port
            for i=1:NoFeatures
                humanKinDynVector(1,i)=human_kin_dynBottle.get(i-1).asFloat64() * scalingVector(1,i);
            end
            
            
            humanKinDynTimeSeries=[humanKinDynTimeSeries;humanKinDynVector]; % all the time series data
            
        end
    end
    pause(sleepTime)
end


disp('human time series data');
disp(humanKinDynTimeSeries);

estimatedRisk=2;

riskVector = risk_port.prepare();
riskVector.clear();

riskVector.push_back(estimatedRisk)
risk_port.write();

disp('estimation is done; closing ...')
pause(1.0);

risk_port.close();
yarp.Network.fini();