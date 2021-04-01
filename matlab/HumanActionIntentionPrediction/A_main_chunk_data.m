clc; clear; close all;
%% DESCRIPTION
%
% This code chunck a long time series data into smaller piecess of time series data.
% This can be used when performing the learning task of time series data and associate 
% the data to training, validation, and test sets.
%
%% PARAMETERS

% enter the list of directories to consider for chunking the data
list_of_root_dirs=[ "../../dataset/HumanActionIntentionPrediction/RawData/Dataset01/"]; 
% e.g.:  ["SynchedData/Session_02/", "SynchedData/Session_03/
% "SynchedData/Session_01/"]
save_file_path = "../../dataset/HumanActionIntentionPrediction/train_val_test_Data";
chunk_length=500;

%% 
files=[];
VariableNames=[];
VariableTypes=[]; 
for i=1:length(list_of_root_dirs)
    rootdir = list_of_root_dirs(i);
    filelist = dir(fullfile(rootdir, '**/*.*'));  %get list of files and folders in any subfolder
    filelist = filelist(~[filelist.isdir]);  %remove folders from list
    for j=1:length(filelist) 
        files=[files, string(strcat(filelist(j).folder, '/',filelist(j).name))];
    end
end

synched_data_list=[];
for i=1:length(files)
    if (isfile(files(i))&& contains(files(i), "Dataset_"))
        [var, VariableNames, VariableTypes ] = importfile(files(i));        
        synched_data={VariableNames, var};
        synched_data_list=[synched_data_list, synched_data];
    end
end

count =0;
chunked_data_list=[];
for i=1:length(synched_data_list)
    data_=synched_data_list(i);
    data_tmp=data_{:,:};
    chunks=1:chunk_length:size(data_tmp,1);
    
    for j= 1:length(chunks)-1
        if j==length(chunks)-1
            samples=chunks(j):size(data_tmp,1);
        else
            samples=chunks(j):chunks(j+1);
        end
                
        chunked_data=data_tmp(samples,:);
        chunked_data_list=[chunked_data_list, {chunked_data}]; 
    end
end

training_percentage= 0.8;
val_percentage=0.1;
test_percentage=0.1;

randomize_data_indices= randperm(length(chunked_data_list));
chunked_data_list_random = chunked_data_list(randomize_data_indices);
training_indices = 1: int64( training_percentage* length(chunked_data_list_random));
val_indices = int64(training_percentage* length(chunked_data_list_random))+1: int64((training_percentage+val_percentage)* length(chunked_data_list_random));
test_indices = int64((training_percentage+val_percentage)* length(chunked_data_list_random)) +1: length(chunked_data_list_random);

training_set=chunked_data_list_random(training_indices);
val_set=chunked_data_list_random(val_indices);
test_set=chunked_data_list_random(test_indices);


% save_data_train_val_test(training_set,val_set, test_set, save_file_path, VariableNames, VariableTypes )

time= var(:,1)-var(1,1);

for i= 2: size(VariableNames,2)
    title_ = strrep(VariableNames(i),'_','-');
    figure;
    plot(time, var(:,i));  title(title_);

end


for i= 2: size(VariableNames,2)
    if contains(VariableNames(i),'Knee')
        VariableNames(i)
        title_ = strrep(VariableNames(i),'_','-');
        figure;
        plot(time, var(:,i));  title(title_);
    end
    
end



          

