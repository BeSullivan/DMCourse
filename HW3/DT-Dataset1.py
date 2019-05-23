import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image 
import pydotplus

col_names = ['ageCode', 'workclassCode', 'fnlwgt', 'educationCode', 'education-num', 'marital-statusCode', 'occupationCode', 'relationshipCode', 'raceCode', 'sexCode', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-countryCode', 'incomeCode']
data = pd.read_csv('Dataset/Transformed1.csv')

feature_cols = ['ageCode', 'workclassCode', 'fnlwgt', 'educationCode', 'education-num', 'marital-statusCode', 'occupationCode', 'relationshipCode', 'raceCode', 'sexCode', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-countryCode']
target = 'incomeCode'
X = data[feature_cols]
Y = data['incomeCode']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

clf = DecisionTreeClassifier(criterion="gini", max_depth=4)

clf = clf.fit(X_train ,Y_train)

Y_pred = clf.predict(X_test)

print("accuracy: ", metrics.accuracy_score(Y_test, Y_pred))

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                feature_names=feature_cols,
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
graph.write_png("DT-Dataset1-Gini.png")