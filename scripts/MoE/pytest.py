import pandas as pd

data_path = '~/element_human-action-intention-recognition/dataset/lifting_test/ergonomic_lifting.txt'

data = pd.read_csv(data_path, sep=' ')

data_mean = data.mean()
data_std = data.std()

print(type(data))
print(data_mean, data_std)