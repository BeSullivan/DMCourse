import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

data = pd.read_csv('data2.csv')

x = 'Amount'

plt.xlabel(x)
plt.hist(data[x])
plt.gcf().savefig('AmountHist.png')
plt.show()
plt.close()


x = ['Units' ,'Day' ,'Amount']

for col in x:  
    plt.xlabel(col)  
    plt.boxplot(data[col])
    plt.gcf().savefig(col + '_BoxPlot.png')
    plt.show()
    plt.close()


x = 'Amount'
y = ['Transaction_Type', 'Week', 'Hour', 'Month']

for i in y:
    plt.xlabel(x)
    plt.ylabel(i)
    plt.scatter(data[x], data[i])
    plt.gcf().savefig(x + '_' + i + '_Scatter.png')
    plt.show()
    plt.close()