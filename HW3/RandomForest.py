import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image 
import pydotplus

col_names = ['ageCode', 'workclassCode', 'fnlwgt', 'educationCode', 'education-num', 'marital-statusCode', 'occupationCode', 'relationshipCode', 'raceCode', 'sexCode', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-countryCode', 'incomeCode']
data = pd.read_csv('Dataset/Transformed1.csv')

feature_cols = ['ageCode', 'workclassCode', 'fnlwgt', 'educationCode', 'education-num', 'marital-statusCode', 'occupationCode', 'relationshipCode', 'raceCode', 'sexCode', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-countryCode']
X = data[feature_cols]
Y = data['incomeCode']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

clfRF = RandomForestClassifier(n_estimators=100, criterion="entropy", max_depth=3)

clfRF.fit(X_train, Y_train)
Y_pred = clfRF.predict(X_test)

print("Dataset1 Accuracy: ", metrics.accuracy_score(Y_test, Y_pred))

# Dataset 2

data = pd.read_csv("Dataset/Transformed2.csv")

feature_cols = ['cap-shapeCode', 'cap-surfaceCode', 'cap-colorCode', 'bruisesCode', 'odorCode', 'gill-attachmentCode', 'gill-spacingCode', 'gill-sizeCode', 'gill-colorCode', 'stalk-shapeCode', 'stalk-rootCode', 'stalk-surface-above-ringCode', 'stalk-surface-below-ringCode', 'stalk-color-above-ringCode', 'stalk-color-below-ringCode', 'veil-typeCode', 'veil-colorCode', 'ring-numberCode', 'ring-typeCode', 'spore-print-colorCode', 'populationCode', 'habitatCode']
target = 'poisonousCode'

X = data[feature_cols]
Y = data[target]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

clfRF.fit(X_train, Y_train)
Y_pred = clfRF.predict(X_test)

print("Dataset2 Accuracy : ", metrics.accuracy_score(Y_test, Y_pred))

# Dataset 3

data = pd.read_csv("Dataset/Dataset3.csv")

feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
target = 'disease'

X = data[feature_cols]
Y = data[target]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

clfRF.fit(X_train, Y_train)
Y_pred = clfRF.predict(X_test)

print("Dataset3 Accuracy : ", metrics.accuracy_score(Y_test, Y_pred))



