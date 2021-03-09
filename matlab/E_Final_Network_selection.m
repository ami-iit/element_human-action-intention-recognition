clc;clear all;close all;
% Data pre processing

%% PARAMETERS

% NN parameters
IterationNo=1; % number of iterations because of random initialization
Min_hiddenLayerSize=20; % min: number of neurons in the hidden layer
Max_hiddenLayerSize=20; % max: number of neurons in the hidden layer

Min_layerDelays=4;     % min: number of recurssion
Max_layerDelays=4;     % min: number of recurssion

timeLengthData= 300;

% Model Data
model_no=4;val_no=1;test_no=1;
experiment_counter=0;

%% LOAD DATA

% load the data
importfile('../dataset/riskAssessment/dataset_37/processedData.mat');

%  Input_NN: 1 X data time length; each cell: No. features X No. Data
%  examples
Input_NN=cell(1,timeLengthData); 

%  Output_NN: 1 X data time length; each cell: No. outputs X No. Data
%  examples
Output_NN=cell(1,timeLengthData);


for i=1:size(processedData,2)
    experiment_counter=experiment_counter+1;
    [jointPos, grf, subjectMass, riskMeasure] = extractData(processedData{i});
    
    %     addRiskAssessmentDataToNN(jointPos, grf, subjectMass, riskMeasure, timeLengthData, experiment_counter, Input_NN, Output_NN);
    jointPos_final= resample(jointPos, timeLengthData,  size(jointPos,1) );
    grf_final= resample(grf, timeLengthData,  size(grf,1) );
    grf_final=grf_final/subjectMass;
    
    input=[jointPos_final, grf_final];
    
    
    for i=1:timeLengthData
        Input_NN{1,i}(:,experiment_counter)=input(i,:)';
        Output_NN{1,i}(:,experiment_counter)=riskMeasure;
        %  multi variable output:
        %     Output_NN{1,data_time}(:,experiment_counter)=[0;1;0;0];
    end
    
end



%%  Test Data
Input_Test=cell(1,351);
Output_Test=cell(1,351);
test_experiment_counter=0;
TESTNo=16:20;
%pickUp
for test_index=TESTNo
    test_experiment_counter=test_experiment_counter+1;
    data_file= fopen(strcat('Data/PickUp/mod (',num2str(test_index),').txt')); 
    data=textscan(data_file,'%f %f %f');fclose(data_file);
    acc(:,1)=data{1};acc(:,2)=data{2};acc(:,3)=data{3};
    for data_time=1:351
        Input_Test{1,data_time}(:,test_experiment_counter)=acc(data_time,:)';
        Output_Test{1,data_time}(:,test_experiment_counter)=[1;0;0;0];
    end    
end
%Screwing
for test_index=TESTNo
    test_experiment_counter=test_experiment_counter+1;
    data_file= fopen(strcat('Data/Screwing/mod (',num2str(test_index),').txt')); 
    data=textscan(data_file,'%f %f %f');fclose(data_file);
    acc(:,1)=data{1};acc(:,2)=data{2};acc(:,3)=data{3};
    for data_time=1:351
        Input_Test{1,data_time}(:,test_experiment_counter)=acc(data_time,:)';
        Output_Test{1,data_time}(:,test_experiment_counter)=[0;1;0;0];
    end    
end
%PutDown
for test_index=TESTNo
    test_experiment_counter=test_experiment_counter+1;
    data_file= fopen(strcat('Data/PutDown/mod (',num2str(test_index),').txt')); 
    data=textscan(data_file,'%f %f %f');fclose(data_file);
    acc(:,1)=data{1};acc(:,2)=data{2};acc(:,3)=data{3};
    for data_time=1:351
        Input_Test{1,data_time}(:,test_experiment_counter)=acc(data_time,:)';
        Output_Test{1,data_time}(:,test_experiment_counter)=[0;0;1;0];
    end    
end
%ScrewingInitial
for test_index=TESTNo
    test_experiment_counter=test_experiment_counter+1;
    data_file= fopen(strcat('Data/ScrewingInitial/mod (',num2str(test_index),').txt')); 
    data=textscan(data_file,'%f %f %f');fclose(data_file);
    acc(:,1)=data{1};acc(:,2)=data{2};acc(:,3)=data{3};
    for data_time=1:351
        Input_Test{1,data_time}(:,test_experiment_counter)=acc(data_time,:)';
Output_Test{1,data_time}(:,test_experiment_counter)=[0;0;0;1];
    end    
end


%%

inputSeries = Input_NN;
targetSeries = Output_NN;

%               pickUp      Screwing    PutDown         ScrewingInitial
TRAINIndex(1,:)=[5:15,      20:30,      35:45,          50:60      ];
TRAINIndex(2,:)=[1:3,8:15	16:18,23:30 31:33,38:45,    46:48,53:60];
TRAINIndex(3,:)=[1:6,11:15  16:21,26:30 31:36,41:45     46:51,56:60];
TRAINIndex(4,:)=[1:9,14,15  16:24,29,30 31:39,44,45     46:54,59,60];
TRAINIndex(5,:)=[1:11,      16:26,      31:41,          46:56];

VALIndex(1,:)=[1,2,         16,17,      31,32,          46,47];
VALIndex(2,:)=[4,5,         19,20,      34,35,          49,50];
VALIndex(3,:)=[7,8,         22,23,      37,38,          52,53];
VALIndex(4,:)=[10,11,       25,26,      40,41,          55,56];
VALIndex(5,:)=[12,13,       27,28,      42,43,          57,58];

TESTIndex(1,:)=[3,4,         18,19,      33,34,          48,49];
TESTIndex(2,:)=[6,7,         21,22,      36,37,          51,52];
TESTIndex(3,:)=[9,10,        24,25,      39,40,          54,55];
TESTIndex(4,:)=[12,13,       27,28,      42,43,          57,58];
TESTIndex(5,:)= [14 15       29 30       44 45           59 60];




%% Structure Selection based on the MSE
data_file= fopen('NetsKFold/Performance_Matrix.txt'); 
data=textscan(data_file,'%u %u %f');fclose(data_file);
COUNTER=data{1};
KFOLDINDEX=data{2};
PERF=data{3};
count=0;

perf1=PERF;
[sortpref,indexpref]=sort(perf1);
selectionNo=30;
i=1:selectionNo;
% Matrix=double([double(LAYERDelaysCounter(indexpref(i))),double(HIDDENLayerSizeCounter(indexpref(i))),double(PREF(indexpref(i)))]);

% DecisionMatrix=zeros(5,10);
% 
% for j1=1:selectionNo
%     k1=LAYERDelaysCounter(indexpref(j1));
%         k2=LayerSize(indexpref(j1));
%         DecisionMatrix(k1,k2)=DecisionMatrix(k1,k2)+1;
% %         if k1==3 && k2==8
% %             counter=COUNTER( indexpref(j1)) 
% %             load(strcat('Nets/net',num2str(counter),'.mat'));                                
% %     end
% end
% % 
% DecisionMatrix(:,1)=1:5;DecisionMatrix(1,:)=1:10;
% sum(sum(DecisionMatrix(2:5,4:10)))      
% DecisionMatrix


%% Structure Selection based on the Weighted MSE
data_file= fopen('NetsKFold/Performance_Matrix.txt'); 
data=textscan(data_file,'%u %u %f');fclose(data_file);
COUNTER=data{1};
KFOLDINDEX=data{2};
PERF=data{3};
count=0;

perf1=PERF;
[sortpref,indexpref]=sort(perf1);
selectionNo=30;
i=1:selectionNo;
% Matrix=double([double(LAYERDelaysCounter(indexpref(i))),double(HIDDENLayerSizeCounter(indexpref(i))),double(PREF(indexpref(i)))]);

% DecisionMatrix=zeros(5,10);
% 
% for j1=1:selectionNo
%     k1=LAYERDelaysCounter(indexpref(j1));
%         k2=LayerSize(indexpref(j1));
%         DecisionMatrix(k1,k2)=DecisionMatrix(k1,k2)+1;
% %         if k1==3 && k2==8
% %             counter=COUNTER( indexpref(j1)) 
% %             load(strcat('Nets/net',num2str(counter),'.mat'));                                
% %     end
% end
% % 
% DecisionMatrix(:,1)=1:5;DecisionMatrix(1,:)=1:10;
% sum(sum(DecisionMatrix(2:5,4:10)))      
% DecisionMatrix

% 
%% Net structure Check

delay=2;layersize=7;
layerDelays=1:delay;
hiddenLayerSize = layersize;
net = layrecnet(layerDelays,hiddenLayerSize);
% for j1=1:30
 j1=24
[Xs,Xi,Ai,Ts] = preparets(net,Input_NN,Output_NN);
counter=COUNTER( indexpref(j1));
KFoldIndex=KFOLDINDEX( indexpref(j1));

trainIndex=TRAINIndex(KFoldIndex,:);
valIndex=VALIndex(KFoldIndex,:);
testIndex =TESTIndex(KFoldIndex,:);

net.divideFcn = 'divideind';
net.divideParam.trainInd =trainIndex;
net.divideParam.valInd = valIndex;
net.divideParam.testInd = testIndex;
net.trainParam.epochs=200;
load(strcat('NetsKFold/net',num2str(counter),'.mat'));
% load(strcat('Nets/net157.mat'));

%         view(net)
        Y = net(Xs,Xi,Ai);
        perf = perform(net,Ts,Y);
%         figure, s=plotresponse(Ts,Y);title(strcat('time series response, counter: ',num2str(counter)));
%         E2 = gsubtract(Ts,Y);
%         figure; ploterrcorr(E2);title(strcat('Autocorrelation , counter: ',num2str(counter)))

        
%         xx=[];
%         for data_time=1:345
%             xx=[ xx Y{1,data_time}(:,testIndex(8))];
%         end
% 
%         figure;
%         plot(xx(1,:)','-b','LineWidth',2); hold on;
%         plot(xx(2,:)','-r','LineWidth',2); hold on;
%         plot(xx(3,:)','-g','LineWidth',2); hold on;
%         plot(xx(4,:)','-k','LineWidth',2); grid on;
%         legend('pickUp','screwing','puDown','initScrew')
%         title('Reponse initScrewing');
%         xx=[];
%         for data_time=1:345
%             xx=[ xx Y{1,data_time}(:,testIndex(6))];
%         end
%         
%         figure;
%         plot(xx(1,:)','-b','LineWidth',2); hold on;
%         plot(xx(2,:)','-r','LineWidth',2); hold on;
%         plot(xx(3,:)','-g','LineWidth',2); hold on;
%         plot(xx(4,:)','-k','LineWidth',2); grid on;
%         legend('pickUp','screwing','puDown','initScrew')
%         title('Reponse tests PutDown');
%        
%          xx=[];
%         for data_time=1:345
%             xx=[ xx Y{1,data_time}(:,testIndex(1))];
%         end
%         
%         figure;
%         plot(xx(1,:)','-b','LineWidth',2); hold on;
%         plot(xx(2,:)','-r','LineWidth',2); hold on;
%         plot(xx(3,:)','-g','LineWidth',2); hold on;
%         plot(xx(4,:)','-k','LineWidth',2); grid on;
%         legend('pickUp','screwing','puDown','initScrew')
%         title('Reponse tests PickUp');
%        
%          xx=[];
%         for data_time=1:345
%             xx=[ xx Y{1,data_time}(:,testIndex(3))];
%         end
%         
%         figure;
%         plot(xx(1,:)','-b','LineWidth',2); hold on;
%         plot(xx(2,:)','-r','LineWidth',2); hold on;
%         plot(xx(3,:)','-g','LineWidth',2); hold on;
%         plot(xx(4,:)','-k','LineWidth',2); grid on;
%         legend('pickUp','screwing','puDown','initScrew')
%         title('Reponse tests Screwing');
       
        %% TEST SET
[XsTest,XiTest,AiTest,TsTest] = preparets(net,Input_Test,Output_Test);
YTest = net(XsTest,XiTest,AiTest);
perfTest = perform(net,TsTest,YTest);

MyPerf= myperfFunc(TsTest,YTest)

 figure, plotresponse(TsTest,YTest);title(strcat('TEST SET: time series response, counter: ',num2str(counter)));
E2Test = gsubtract(TsTest,YTest);
figure; ploterrcorr(E2Test);title(strcat('TEST SET: Autocorrelation , counter: ',num2str(counter)))
for i=1:20
xYTest=[];xTsTest=[];
for data_time=1:345
    xYTest=    [ xYTest YTest{1,data_time}(:,i)];
    xTsTest=   [xTsTest TsTest{1,data_time}(:,i)];
end
        
        figure, plotresponse(xTsTest,xYTest);title(strcat('2222TEST SET: time series response, counter: ',num2str(counter)));
        figure;
        plot(xYTest(1,:)','-b','LineWidth',2); hold on;
        plot(xYTest(2,:)','-r','LineWidth',2); hold on;
        plot(xYTest(3,:)','-g','LineWidth',2); hold on;
        plot(xYTest(4,:)','-k','LineWidth',2); 
        legend('pickUp','screwing','puDown','initScrew')
        title(['TEST SET Result:',num2str(i)]);
end

% Confusion Matrix, Accuracy:
lastInstantTest = YTest{end};
lastInstantTs = TsTest{end};
ClassResTest=round(lastInstantTest(:,:));

[c,cm,ind,per] = confusion(lastInstantTs,lastInstantTest);
 plotconfusion(lastInstantTs,lastInstantTest)
accuracy(j1)=1-c;

clc;
% end
[sortACC,indexACC]=sort(accuracy,'descend');

