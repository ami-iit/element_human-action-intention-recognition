function []= save_data_train_val_test(training_set,val_set, test_set, file_path, VariableNames, VariableTypes)
formatSpec_names=[];
formatSpec_values=[];

for i=1:size(VariableNames,2)
    variables(i)=convertCharsToStrings(VariableNames{i});
    if i<size(VariableNames,2)
        formatSpec_names=[formatSpec_names,'%s '];
        if i==1
            formatSpec_values=[formatSpec_values,'%.6f '];
        else 
            formatSpec_values=[formatSpec_values,'%d '];
        end
    else
        formatSpec_names=[formatSpec_names,'%s\n'];
        formatSpec_values=[formatSpec_values,'%d\n'];
    end
end

% formatSpec_names ='%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n';
% formatSpec_values ='%.6f %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %.6f\n';


%% save training set
data_type='/training';
save_file_path=strcat(file_path, data_type);
data_set=training_set;

if ~exist(save_file_path, 'dir')
    mkdir(save_file_path)
end

for i=1:length(data_set)
    values=data_set{i};
    fileID = fopen(strcat(save_file_path, data_type , string(i),'.txt'),'w');
    fprintf(fileID,formatSpec_names,variables);
    fprintf(fileID,formatSpec_values,values');
    fclose(fileID);
end

%% save validation set
data_type='/validation';
data_set=val_set;

save_file_path=strcat(file_path, data_type);

if ~exist(save_file_path, 'dir')
    mkdir(save_file_path)
end

for i=1:length(data_set)
    values=data_set{i};
    fileID = fopen(strcat(save_file_path, data_type , string(i),'.txt'),'w');
    fprintf(fileID,formatSpec_names,variables);
    fprintf(fileID,formatSpec_values,values');
    fclose(fileID);
end

%% save test set
data_type='/test';
data_set=test_set;

save_file_path=strcat(file_path, data_type);

if ~exist(save_file_path, 'dir')
    mkdir(save_file_path)
end

for i=1:length(data_set)
    values=data_set{i};
    fileID = fopen(strcat(save_file_path, data_type , string(i),'.txt'),'w');
    fprintf(fileID,formatSpec_names,variables);
    fprintf(fileID,formatSpec_values,values');
    fclose(fileID);
end

end
