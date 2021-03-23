function addRiskAssessmentDataToNN(jointPos, grf, subjectMass, riskMeasure, timeLengthData, experiment_counter, Input_NN, Output_NN)


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
