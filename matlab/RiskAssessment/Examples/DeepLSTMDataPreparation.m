function [XTrain,YTrain, XTest,YTest]= DeepLSTMDataPreparation(dirDataset, testSetPercentage, classification)

training_data_counter=0;
test_data_counter=0;
lifting_action_durations=[];

%  data types: 1 X data time length; each cell: No. features X No. Data examples
XTrain={};
YTrain=[];

XTest ={};
YTest =[];

subjctNames= GetSubDirsFirstLevelOnly(dirDataset);

for j=1:size(subjctNames,2)-1
    trialName = GetSubDirsFirstLevelOnly(strcat(dirDataset,'/',subjctNames{j},'/processed'));    
    for i = 1:size(trialName,2)
        infoFileName = strcat(dirDataset,'/',subjctNames{j},'/processed/',trialName{i},'/info.mat');
        info = importdata(infoFileName);
        kinematicFileName = strcat(dirDataset,'/',subjctNames{j},'/processed/',trialName{i},'/kinematicData.mat');
        kinematicData = importdata(kinematicFileName);
        wrenchFileName = strcat(dirDataset,'/',subjctNames{j},'/processed/',trialName{i},'/measurements.mat');
        wrenchData = importdata(wrenchFileName);
        
        if kinematicData.ikSolved
            start_indx = info.startSampleIndex;
            end_indx= info.endSampleIndex;
            subjectWeight = info.mass*9.81;
            riskMeasure = info.LI;           
            jointPos = kinematicData.jointPos_raw(start_indx:end_indx,:);
            grf = [ wrenchData.wrenchData.LeftFoot(start_indx:end_indx,:), ...
                wrenchData.wrenchData.RightFoot(start_indx:end_indx,:)]; % ground reaction wrenches 
            
            lifting_action_time = (end_indx-start_indx)/240;
            lifting_action_durations=[lifting_action_durations, lifting_action_time];
            
   %         jointPos_final= resample(jointPos, timeLengthData,  size(jointPos,1) );
   %         grf_final= resample(grf, timeLengthData,  size(grf,1) );
            jointPos_final= jointPos;
            grf_final= grf;
            grf_final=grf_final/subjectWeight; % CHECK THIS LATER
            
            input=[jointPos_final, grf_final];
            
            % test set
            if rand < testSetPercentage
                test_data_counter= test_data_counter+1;
                
                    XTest{end+1,1}=input';
                    if classification==true
                        %	multi variable output:
                        %output=[0;0;0];
                        %output(riskMeasure)=1;
                        YTest(end+1,1)=riskMeasure;
                    else
                        YTest(end+1,1)=riskMeasure;
                    end
                    


            else % training and validation set
                training_data_counter=training_data_counter+1;
                
                    XTrain{end+1,1}=input';
                    if classification==true
                        %	multi variable output:
                        %output=[0;0;0];
                        %output(riskMeasure)=1;
                        YTrain(end+1,1)=riskMeasure;
                    else
                        YTrain(end+1,1)=riskMeasure;
                    end

            end
        end
    end
end
if classification==true
    YTrain=categorical(YTrain, [1 2 3]);
    YTest=categorical(YTest, [1 2 3]);

end

end