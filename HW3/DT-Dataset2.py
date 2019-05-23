import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image 
import pydotplus

data = pd.read_csv("Dataset/Transformed2.csv")

feature_cols = ['cap-shapeCode', 'cap-surfaceCode', 'cap-colorCode', 'bruisesCode', 'odorCode', 'gill-attachmentCode', 'gill-spacingCode', 'gill-sizeCode', 'gill-colorCode', 'stalk-shapeCode', 'stalk-rootCode', 'stalk-surface-above-ringCode', 'stalk-surface-below-ringCode', 'stalk-color-above-ringCode', 'stalk-color-below-ringCode', 'veil-typeCode', 'veil-colorCode', 'ring-numberCode', 'ring-typeCode', 'spore-print-colorCode', 'populationCode', 'habitatCode']
target = 'poisonousCode'

X = data[feature_cols]
Y = data[target]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

clf = DecisionTreeClassifier(criterion="gini", max_depth=4)

clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)

print("accuracy : ", metrics.accuracy_score(Y_test, Y_pred))

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                feature_names=feature_cols,
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
graph.write_png("DT-Dataset2-Gini.png")