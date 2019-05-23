import pandas as pd 
from sklearn import preprocessing

data = pd.read_csv("Dataset/Dataset1.csv")


agebins = [0 ,10 ,20 ,30 ,40 ,50 ,200]
labels = ['child', 'teenager', 'young', 'adult', 'middle-aged', 'old']

data['age'] = pd.cut(data['age'], bins=agebins, labels=labels, right=False)

catHeaders = ['age', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income']

enc = preprocessing.LabelEncoder()

for i in catHeaders:
    data[i+'Code'] = enc.fit_transform(data[i])

print (data)

data.to_csv('Dataset/Transformed1.csv')


