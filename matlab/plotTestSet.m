function plotTestSet(TsTest,YTest)

noClasses= size(TsTest{1},1); % in a classification problem this is bigger than "1".

for i=1:size(TsTest{1},2)
    for k= 1:noClasses
        y_ground_truth={};
        y_estimated={};
        
        for j=1:size(TsTest,2)
            y_ground_truth{j}= TsTest{j}(k,i); % TsTest
            y_estimated{j}= YTest{j}(k,i); % YTest
        end
        
        figure;
        plotresponse(y_ground_truth,y_estimated);
        title(strcat('test data (',num2str(i),'-',num2str(k),'): target and estimated risk'));
        saveas(gcf,strcat('E',num2str(i),'-',num2str(k),'.jpg'))
    end
    
end

end