from abc import ABC, abstractmethod
import numpy as np
import os
import io


class Dataset(ABC):
    """
      Abstract Class for Sequential Dataset
    """
    def __init__(self):
        return

    def get_list_of_files_in_dir(self, dir_name):
        # create a list of file and sub directories
        # names in the given directory
        list_of_file = os.listdir(dir_name)
        all_files = list()
        # Iterate over all the entries
        for entry in list_of_file:
            # Create full path
            full_path = os.path.join(dir_name, entry)
            # If entry is a directory then get the list of files in this directory
            if os.path.isdir(full_path):
                all_files = all_files + self.get_list_of_files_in_dir(full_path)
            else:
                all_files.append(full_path)

        return all_files

    def read_data_in_directory(self, dir_path):
        print('[DataSet:read_data_in_directory] directory path: ', dir_path)
        files = self.get_list_of_files_in_dir(dir_path)
        # files = [f for f in listdir(file_path) if isfile(join(file_path, f))]
        print('[DataSet:read_data_in_directory] files: ', files)
        data_raw = []
        for file in files:
            # file_name = os.join(file_path, file)
            print('[DataSet:read_data_in_directory] file: ', file)
            data_dict = self.read_data_from_file(file)
            data_raw.append(data_dict)
        return data_raw

    def read_data_from_file(self, file):
        print('[DataSet:read_data_from_file] file: ', file)
        data_dict = {}
        input_file = io.open(file, 'r')
        # lines = input_file.readlines()
        count = 0
        keys_list = []
        while True:
            count += 1
            # Get next line from file
            line = input_file.readline()
            # if line is empty
            # end of file is reached
            if not line:
                break

            line_elements = line.split()
            i = 0
            for element in line_elements:
                if count == 1:
                    data_dict[element] = []
                    keys_list.append(element)
                else:
                    element = float(element)
                    data_dict[keys_list[i]].append(element)
                i += 1
        input_file.close()
        return data_dict

    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod
    def plot_data(self):
       pass

