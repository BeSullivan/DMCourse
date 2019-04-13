import pandas as pd 

data = pd.read_csv('data.csv')

for column in data:
    print(column ,end='\t')

print('\nSize of data', data.size)
print('Number of rows', len(data))