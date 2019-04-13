import pandas as pd 
import numpy as np 

data = pd.read_csv('data2.csv')

print('Max of Amount : ', np.max(data['Amount']))
print('Min of Amount : ', np.min(data['Amount']))
print('Mean of Amount : ', np.median(data['Amount']))
print('Average of Amount : ', np.mean(data['Amount']))
print('Std of Amount : ', np.std(data['Amount']))