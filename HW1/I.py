import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

data = pd.read_csv('data2.csv')

x = data['Amount']

std_amount = np.std(x)
mean_amount = np.mean(x)

data = data[ mean_amount - 3 * std_amount < data['Amount'] ]
data = data[ data['Amount'] < mean_amount + 3 * std_amount ]

plt.boxplot(data['Amount'])
plt.show()