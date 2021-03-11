function [Input_NN, Output_NN, Input_Test, Output_Test] = prepareDataset(dirDataset, timeLengthData, testSetPercentage)
training_data_counter=0;
test_data_counter=0;

%  data types: 1 X data time length; each cell: No. features X No. Data examples
Input_NN=cell(1,timeLengthData);
Output_NN=cell(1,timeLengthData);
Input_Test=cell(1,timeLengthData);
Output_Test=cell(1,timeLengthData);

dirNames= GetSubDirsFirstLevelOnly(dirDataset);

for j=1:size(dirNames,2)-1
    fileName = strcat(dirDataset,'/',dirNames{j},'/processed/processedData.mat');
    data = importdata(fileName);
    
    for i = 1:size(data,2)
        if data{i}.kinematicData.ikSolved
            [jointPos, grf, subjectMass, riskMeasure] = extractData(data{i});
            
            jointPos_final= resample(jointPos, timeLengthData,  size(jointPos,1) );
            grf_final= resample(grf, timeLengthData,  size(grf,1) );
            grf_final=grf_final/subjectMass; % CHECK THIS LATER
            
            input=[jointPos_final, grf_final];
            
            % test set
            if rand < testSetPercentage
                test_data_counter= test_data_counter+1;
                for k=1:timeLengthData%% Test Data
                    
                    Input_Test{1,k}(:,test_data_counter)=input(k,:)';
                    Output_Test{1,k}(:,test_data_counter)=riskMeasure;
                    %  multi variable output:
                    %     Output_NN{1,data_time}(:,experiment_counter)=[0;1;0;0];
                end
            else % training and validation set
                training_data_counter=training_data_counter+1;
                
                for k=1:timeLengthData
                    Input_NN{1,k}(:,training_data_counter)=input(k,:)';
                    Output_NN{1,k}(:,training_data_counter)=riskMeasure;
                end
            end
        end
    end
end

end

