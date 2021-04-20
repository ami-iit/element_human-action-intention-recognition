function [inputDataPrepared] = prepareDatasetOnline(inputData, timeLengthData)

%  data types: 1 X data time length; each cell: No. features X No. Data examples
Input_NN=cell(1,timeLengthData);
Output_NN=cell(1,timeLengthData);
inputDataPrepared=cell(1,timeLengthData);
Output_Test=cell(1,timeLengthData);

inputDataFinal= resample(inputData, timeLengthData,  size(inputData,1) );

for k=1:timeLengthData%% Test Data
    inputDataPrepared{1,k}(:,1)=inputDataFinal(k,:)';
end


end

