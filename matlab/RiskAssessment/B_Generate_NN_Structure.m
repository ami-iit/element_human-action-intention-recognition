%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%        Generate Different NN structures with loop:                            %%%%%%%
%%%%        No. of Neurons (hidden layer size) *No.  Delays * No. of iteration     %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



clc;
clear;
close all;

%% PARAMETERS

% NN parameters
NN_Name='Nets/netRegression3-';
IterationNo=4; % number of iterations because of random initialization
Min_hiddenLayerSize=5; % min: number of neurons in the hidden layer
step_hiddenLayerSize=10;
Max_hiddenLayerSize=55; % max: number of neurons in the hidden layer

Min_layerDelays=2;     % min: number of recurssion
Max_layerDelays=3;     % min: number of recurssion

timeLengthData= 50;
% percentage of the test set among all the data
testSetPercentage=0.1;
% percentage of the validation set in the data used for training
validationPercentage= 0.1;

% Model Data
model_no=4;val_no=1;test_no=1;

classification= false;
saveModel=true;

%% LOAD DATA

% load the data

dirDataset = '../../../DataSet/RiskAssessment/processed_lifting_data/processed_lifting_data';
[Input_NN, Output_NN, Input_Test, Output_Test] = prepareDataset(dirDataset, timeLengthData, testSetPercentage, classification);

%% NN

% Solve an Autoregression Problem with External Input with a NARX Neural Network
%
% This script assumes these variables are defined:
%
%   Input_NN - input time series.
%   Output_NN - feedback time series.

inputSeries = Input_NN;
targetSeries = Output_NN;

%inputDelays = 1:2;
%feedbackDelays = 1:3;

% loops:  hiddenLayerSize , layerDelays, trainIndex ,*10 for check
% 4-15 * 2-5 * 4 * 10 --> 1920 times (1 min) --> 32 hr
% make list of nets, 
% 4-12 * 2-5 * 3 * 10 --> 1080 times (1 min) --> 18 hr

% single k-fold labels, 
% trainIndex=[1,2,4];
% valIndex=[3];
% testIndex = [];

% randomly select training and validation set
array_size=size(Input_NN{1},2);
rnd_indices=randperm(array_size, array_size);
valIndex=rnd_indices(1:round(array_size*validationPercentage));
trainIndex=rnd_indices(round(array_size*validationPercentage)+1:end);
testIndex = [];


% For finding proper structure we dont do the k-fold cross validation.

% % Min_hiddenLayerSize=4;  Max_hiddenLayerSize=12;
% % Min_layerDelays=2;      Max_layerDelays=5;


% Totally: 10*9*4 NN.
counter=0;
Performance_Matrix=[];

for layerDelaysCounter=Min_layerDelays:Max_layerDelays
for hiddenLayerSizeCounter =Min_hiddenLayerSize:step_hiddenLayerSize:Max_hiddenLayerSize
for iterator=1:IterationNo
    
counter=counter+1
layerDelays=1:layerDelaysCounter;
hiddenLayerSize = hiddenLayerSizeCounter;


%net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize);
net = layrecnet(layerDelays,hiddenLayerSize);

% Choose Input and Feedback Pre/Post-Processing Functions
% Settings for feedback input are automatically applied to feedback output
% For a list of all processing functions type: help nnprocess
% Customize input parameters at: net.inputs{i}.processParam
% Customize output parameters at: net.outputs{i}.processParam
%- net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
%- net.inputs{2}.processFcns = {'removeconstantrows','mapminmax'};

% Prepare the Data for T(:,test_data_counter)raining and Simulation
% The function PREPARETS prepares timeseries data for a particular network,
% shifting time by the minimum amount to fill input states and layer states.
% Using PREPARETS allows you to keep your original time series data unchanged, while
% easily customizing it for networks with differing numbers of delays, with
% open loop or closed loop feedback modes.
%- [inputs,inputStates,layerStates,targets] = preparets(net,inputSeries,{},targetSeries);

[Xs,Xi,Ai,Ts] = preparets(net,Input_NN,Output_NN);

% Setup Division of Data for Training, Validation, Testing
% The function DIVIDERAND randomly assigns target values to training,
% validation and test sets during training.
% For a list of all data division functions type: help nndivide
%- net.divideFcn = 'dividerand';  % Divide data randomly
% The property DIVIDEMODE set to TIMESTEP means that targets are divided
% into training, validation and test sets according to timesteps.
% For a list of data division modes type: help nntype_data_division_mode
%- net.divideMode = 'value';  % Divide up every value
%- net.divideParam.trainRatio = 70/100;
%- net.divideParam.valRatio = 15/100;
%- net.divideParam.testRatio = 15/100;

net.divideFcn = 'divideind';
net.divideParam.trainInd =trainIndex;
net.divideParam.valInd = valIndex;
net.divideParam.testInd = testIndex;

% net.layers{2}.transferFcn='logsig';

% Choose a Training Function
% For a list of all training functions type: help nntrain
% Customize training parameters at: net.trainParam
%- net.trainFcn = 'trainlm';  % Levenberg-Marquardt, %by default is this
%option

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
% Customize performance parameters at: net.performParam
%- net.performFcn = 'mse';  % Mean squared error

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
% Customize plot parameters at: net.plotParam
%- net.plotFcns = {'plotperform','plottrainstate','plotresponse', ...
%-  'ploterrcorr', 'plotinerrcorr'};

% Train the Network
% net: New networktest: response
 
% tr: Training record (epoch and perf)
 

%- [net,tr] = train(net,inputs,targets,inputStates,layerStates);

% Xs Shifted inputs
% TsShifted targets
% XiInitial input delay states
% AiInitial layer delay states

[net,tr ]= train(net,Xs,Ts,Xi,Ai);
%load('Nets/net6.mat')

% Test the Network
%- outputs = net(inputs,inputStates,layerStates);
%- errors = gsubtract(targets,outputs);
%- performance = perform(net,targets,outputs)
%% Training Set Test

Y = net(Xs,Xi,Ai);
perf = perform(net,Ts,Y)

% Recalculate Training, Validation and Test Performance
%- trainTargets = gmultiply(targets,tr.trainMask);
%- valTargets = gmultiply(targets,tr.valMask);
%- testTargets = gmultiply(targets,tr.testMask);
%- trainPerformance = perform(net,trainTargets,outputs)
%- valPerformance = perform(net,valTargets,outputs)
%- testPerformance = perform(net,testTargets,outputs)

% View the Network
%- view(net)



% Closed Loop Network
% Use this network to do multi-step prediction.
% The function CLOSELOOP replaces the feedback input with a direct
% connection from the outout layer.
%- netc = closeloop(net);
%- netc.name = [net.name ' - Closed Loop'];
%- view(netc)
%- [xc,xic,aic,tc] = preparets(netc,inputSeries,{},targetSeries);
%- yc = netc(xc,xic,aic);
%- closedLoopPerformance = perform(netc,tc,yc)

% Early Prediction Network
% For some applications it helps to get the prediction a timestep early.
% The original network returns predicted y(t+1) at the same time it is given y(t+1).
% For some applications such as decision making, it would help to have predicted
% y(t+1) once y(t) is available, but before the actual y(t+1) occurs.
% The network can be made to return its output a timestep early by removing one delay
% so that its minimal tap delay is now 0 instead of 1.  The new network returns the
% same outputs as the original network, but outputs are shifted left one timestep.
%- nets = removedelay(net);
%- nets.name = [net.name ' - Predict One Step Ahead'];
%- view(nets)
%- [xs,xis,ais,ts] = preparets(nets,inputSeries,{},targetSeries);
%- ys = nets(xs,xis,ais);
%- earlyPredictPerformance = perform(nets,ts,ys)


%%lastInstant = Y{end};test: response
%%ClassRes=round(lastInstant(:,testIndex));




%% Test set

[XsTest,XiTest,AiTest,TsTest] = preparets(net,Input_Test,Output_Test);
YTest = net(XsTest,XiTest,AiTest);
perfTest = perform(net,TsTest,YTest);

% Recalculate Training, Validation and Test Performance
%- trainTargets = gmultiply(targets,tr.trainMask);
%- valTargets = gmultiply(targets,tr.valMask);
%- testTargets = gmultiply(targets,tr.testMask);
%- trainPerformance = perform(net,trainTargets,outputs)
%- valPerformance = perform(net,valTargets,outputs)
%- testPerformance = perform(net,testTargets,outputs)

% View the Network
%- view(net)


% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, plotregression(targets,outputs)
%figure, plotresponse(targets,outputs)
%figure, ploterrcorr(errors)
%figure, plotinerrcorr(inputs,errors)

% Closed Loop Network
% Use this network to do multi-step prediction.
% The function CLOSELOOP replaces the feedback input with a direct
% connection from the outout layer.
%- netc = closeloop(net);
%- netc.name = [net.name ' - Closed Loop'];
%- view(netc)
%- [xc,xic,aic,tc] = preparets(netc,inputSeries,{},targetSeries);
%- yc = netc(xc,xic,aic);
%- closedLoopPerformance = perform(netc,tc,yc)

% Early Prediction Network
% For some applications it helps to get the prediction a timestep early.
% The original network returns predicted y(t+1) at the same time it is given y(t+1).
% For some applications such as decision making, it would help to have predicted
% y(t+1) once y(t) is available, but before the actual y(t+1) occurs.
% The network can be made to return its output a timestep early by removing one delay
% so that its minimal tap delay is now 0 instead of 1.  The new network returns the
% same outputs as the original network, but outputs are shifted left one timestep.
%- nets = removedelay(net);
%- nets.name = [net.name ' - Predict One Step Ahead'];
%- view(nets)
%- [xs,xis,ais,ts] = preparets(nets,inputSeries,{},targetSeries);
%- ys = nets(xs,xis,ais);
%- earlyPredictPerformance = perform(nets,ts,ys)


% % lastInstantTest = YTest{end};
% % ClassResTest=round(lastInstantTest(:,:))
% % 
% % xx=[];
% % for data_time=1:346
% % xx=[ xx YTest{1,data_time}(:,20)];
% % end 
% % figure;plot(xx');title('InitScrewing');


%% Post Processing:

%ClassDesRes=zeros(4,8);
%ClassDesRes(1,1:2)=1;ClassDesRes(2,3:4)=1;ClassDesRes(3,5:6)=1;ClassDesRes(4,7:8)=1;

%% Save and Load;
% Net2=net;
% save Net2;
%load Net1;

if saveModel
    %mkdir('Nets/net1')
    savefile = strcat(NN_Name,num2str(counter),'.mat');
    save(savefile ,'net')
end

% how increase number of layers and check it
% increase number of layers --> 15 for modeling as GMM
% k-fold analysis

% with diff number of neurons and delays in another computer
% test the experiments maybe by simulink
% add more data of the actions --> each 20


%% plots
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, plotregression(targets,outputs)
%figure, plotresponse(targets,outputs)
%figure, ploterrcorr(errors)
%figure, plotinerrcorr(inputs,errors)


figure, 
plotperform(tr);
saveas(gcf,strcat('Figures/plotperform',num2str(counter),'.jpg'))

figure; plotresponse(TsTest,YTest);title('test: response');%saveas(gcf,'E1.jpg')

plotTestSet(TsTest,YTest);
% E1 = gsubtract(TsTest,YTest);
% figure; ploterrcorr(E1);title('test: errcorr');saveas(gcf,'E2.jpg')
% 
% figure, plotresponse(Ts,Y)
% E2 = gsubtract(Ts,Y);
% figure; ploterrcorr(E2);saveas(gcf,'E3.jpg')

Performance_Matrix(:,counter)=[counter;layerDelaysCounter;hiddenLayerSizeCounter;perf ];

close all;

end
end
end

figure; plot(Performance_Matrix(1,:),Performance_Matrix(4,:),'-*k');
title('Performance of Neural Networks Models');
saveas(gcf,'NN1.jpg');

if classification==true
    lastInstantTest = YTest{end};
    lastInstantTs = TsTest{end};
    ClassResTest=round(lastInstantTest(:,:));

    [c,cm,ind,per] = confusion(lastInstantTs,lastInstantTest);
    plotconfusion(lastInstantTs,lastInstantTest)
    accuracy=1-c
    saveas(gcf,'confusionMatrix.jpg')
end
