function plotTestSet(TsTest,YTest)


for i=1:size(TsTest{1},2)
    y_ground_truth={};
    y_estimated={};
    
    for j=1:size(TsTest,2)
        y_ground_truth{j}= TsTest{j}(:,i); % TsTest
        y_estimated{j}= YTest{j}(:,i); % YTest
    end
    
    figure; 
    plotresponse(y_ground_truth,y_estimated);
    title(strcat('test data (',num2str(i),'): target and estimated risk'));
    saveas(gcf,strcat('E',num2str(i),'.jpg'))

end





end