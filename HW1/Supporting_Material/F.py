import pandas as pd 
import numpy as np 

data = pd.read_csv('data2.csv')

print('Fields in Amount column :', data['Amount'].count(), end='\n\n')

print(data.count())