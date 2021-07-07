clc;  close all;
% clear;
% the code is broken/incomplete

%% Useful info
% matlab command for an example of the prediction scenario:
%>> openExample('nnet/TimeSeriesForecastingUsingDeepLearningExample')


%% PARAMETERS

NN_Name='Nets/netclassDeepLSTM-';

testSetPercentage=0.2;

classification= true;

%% DATA PREPARATION
dirDataset = '../../../DataSet/RiskAssessment/processed_lifting_data/processed_lifting_data';
[XTrain,YTrain, XValidation, YValidation] = DeepLSTMDataPreparation(dirDataset, testSetPercentage, classification);

miniBatchSize = 27;
options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'MaxEpochs',100, ...
    'MiniBatchSize',miniBatchSize, ...
    'ValidationData',{XValidation,YValidation}, ...
    'GradientThreshold',2, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress');
net = trainNetwork(XTrain,YTrain,layers,options);


YPred = classify(net,XValidation,'MiniBatchSize',miniBatchSize);
acc = mean(YPred == YValidation)
