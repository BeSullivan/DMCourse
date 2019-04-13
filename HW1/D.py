import pandas as pd

data = pd.read_csv('data2.csv')

data = data.drop(columns='BranchName')

print(data.sample(10))

data.to_csv('data2.csv',index=False)