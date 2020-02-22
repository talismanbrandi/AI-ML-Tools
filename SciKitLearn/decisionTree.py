import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
import pydotplus

""" Read data for fitting
"""
data = pd.read_csv('data/Social_Network_Ads.csv')
data.head()

""" Split into dependent and independent variables
"""
feature_columns = ['Age', 'EstimatedSalary']
X = data.iloc[:,[2,3]].values
Y = data.iloc[:, 4].values

""" Split for training and testing
"""
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

""" Normalize data
"""
scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

""" Fit the decision tree
"""
classifier = DecisionTreeClassifier()
classifier = classifier.fit(X_train, Y_train)

""" Predictions
"""
Y_pred = classifier.predict(X_test)
print('Accuracy Score: ', metrics.accuracy_score(Y_test, Y_pred))

""" The confusion matrix
"""
cm = metrics.confusion_matrix(Y_test, Y_pred)

""" Make the contour plot
"""
X_set, Y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min()-5, stop=X_set[:, 0].max()+5, step=1),
                     np.arange(start=X_set[:, 1].min()-5000, stop=X_set[:, 1].max()+5000, step=10))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha=0.75,
             cmap=ListedColormap(("#8b0200", "#00a1a4")))

for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1], c=ListedColormap(("#570100", "#006f71"))(i), label=j)
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

""" Print the tree
"""
dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=feature_columns,
                class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("fullTree.png")

""" Refit with pruning
"""
classifier=DecisionTreeClassifier(criterion='entropy', max_depth=3)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
print('Accuracy Score: ', metrics.accuracy_score(Y_test, Y_pred))

""" Print the new tree
"""
dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=feature_columns,
                class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("prunedTree.png")