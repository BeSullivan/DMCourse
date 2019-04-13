import pandas as pd 

data = pd.read_csv('data.csv')

print(data, end = '\n\n')

data['Amount'] = data['Amount'].apply( lambda amount : amount.replace('$', '').replace(',', '') )

print(data)

data.to_csv('data2.csv', index=False)