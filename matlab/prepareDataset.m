function [Input_NN, Output_NN, Input_Test, Output_Test] = prepareDataset(processedData, timeLengthData)

experiment_counter=0;

%  Input_NN: 1 X data time length; each cell: No. features X No. Data
%  examples
Input_NN=cell(1,timeLengthData);

%  Output_NN: 1 X data time length; each cell: No. outputs X No. Data
%  examples
Output_NN=cell(1,timeLengthData);


for i=1:size(processedData,2)-1
    experiment_counter=experiment_counter+1;
    [jointPos, grf, subjectMass, riskMeasure] = extractData(processedData{i});
    
    %     addRiskAssessmentDataToNN(jointPos, grf, subjectMass, riskMeasure, timeLengthData, experiment_counter, Input_NN, Output_NN);
    jointPos_final= resample(jointPos, timeLengthData,  size(jointPos,1) );
    grf_final= resample(grf, timeLengthData,  size(grf,1) );
    grf_final=grf_final/subjectMass;
    
    input=[jointPos_final, grf_final];
    
    
    for k=1:timeLengthData
        Input_NN{1,k}(:,experiment_counter)=input(k,:)';
        Output_NN{1,k}(:,experiment_counter)=riskMeasure;
        %  multi variable output:
        %     Output_NN{1,data_time}(:,experiment_counter)=[0;1;0;0];
    end
    
end


%% Test Data

Input_Test=cell(1,timeLengthData);
Output_Test=cell(1,timeLengthData);

experiment_counter=0;
for i=size(processedData,2):size(processedData,2)
    experiment_counter=experiment_counter+1;
    [jointPos, grf, subjectMass, riskMeasure] = extractData(processedData{i});
    
    %     addRiskAssessmentDataToNN(jointPos, grf, subjectMass, riskMeasure, timeLengthData, experiment_counter, Input_NN, Output_NN);
    jointPos_final= resample(jointPos, timeLengthData,  size(jointPos,1) );
    grf_final= resample(grf, timeLengthData,  size(grf,1) );
    grf_final=grf_final/subjectMass;
    
    input=[jointPos_final, grf_final];
    
    
    for k=1:timeLengthData
        Input_Test{1,k}(:,experiment_counter)=input(k,:)';
        Output_Test{1,k}(:,experiment_counter)=riskMeasure;
        %  multi variable output:
        %     Output_NN{1,data_time}(:,experiment_counter)=[0;1;0;0];
    end
    
end

end

