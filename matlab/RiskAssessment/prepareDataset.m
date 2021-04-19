function [Input_NN, Output_NN, Input_Test, Output_Test] = prepareDataset(dirDataset, timeLengthData, testSetPercentage, classification)
training_data_counter=0;
test_data_counter=0;
lifting_action_durations=[];

%  data types: 1 X data time length; each cell: No. features X No. Data examples
Input_NN=cell(1,timeLengthData);
Output_NN=cell(1,timeLengthData);
Input_Test=cell(1,timeLengthData);
Output_Test=cell(1,timeLengthData);

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
            
            jointPos_final= resample(jointPos, timeLengthData,  size(jointPos,1) );
            grf_final= resample(grf, timeLengthData,  size(grf,1) );
            grf_final=grf_final/subjectWeight; % CHECK THIS LATER
            
            input=[jointPos_final, grf_final];
            
            % test set
            if rand < testSetPercentage
                test_data_counter= test_data_counter+1;
                for k=1:timeLengthData%% Test Data
                    
                    Input_Test{1,k}(:,test_data_counter)=input(k,:)';
                    if classification==true
                        %	multi variable output:
                        output=[0;0;0];
                        output(riskMeasure)=1;
                        Output_Test{1,k}(:,test_data_counter)=output;
                    else
                        Output_Test{1,k}(:,test_data_counter)=riskMeasure;
                    end
                    

                end
            else % training and validation set
                training_data_counter=training_data_counter+1;
                
                for k=1:timeLengthData
                    Input_NN{1,k}(:,training_data_counter)=input(k,:)';
                    if classification==true
                        %	multi variable output:
                        output=[0;0;0];
                        output(riskMeasure)=1;
                        Output_NN{1,k}(:,training_data_counter)=output;
                    else
                        Output_NN{1,k}(:,training_data_counter)=riskMeasure;
                    end
                end
            end
        end
    end
end

end

