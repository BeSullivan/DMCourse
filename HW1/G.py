import pandas as pd
import numpy as np 

data = pd.read_csv('data2.csv')

print(data.sample(10).isnull(), end='\n\n')

print(data.isnull().any())

