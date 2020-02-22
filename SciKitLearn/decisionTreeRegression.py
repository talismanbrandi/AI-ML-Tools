import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.externals.six import StringIO
import pydotplus
import math as m

""" Initialize some variables and containers
"""
nx = 1000
nx2 = int(nx/2)
printTree = False

xa = np.zeros((nx, 2))
X_train = np.zeros(nx)
Y_train = np.zeros(nx)
X_test = np.zeros(nx2)
Y_test = np.zeros(nx2)

""" Generate data for fitting
"""
for i in range (0, nx):
    X_train[i] = i * 2. * m.pi/(nx - 1.) + 0.01
    Y_train[i] = m.sin(X_train[i]) * m.cos(4.*X_train[i]) * (1. + np.random.normal(0., 0.2, 1))

for i in range(0, nx2):
    X_test[i] = i * 2. * m.pi / (nx2 - 1.) + 0.01
    Y_test[i] = m.sin(X_test[i]) * m.cos(4. * X_test[i])

""" Fit the decision tree
"""
regressor = tree.DecisionTreeRegressor(splitter='random', min_samples_split=10)
regressor = regressor.fit(X_train.reshape(-1, 1), Y_train)

""" Test the tree
"""
Y_pred = regressor.predict(X_test.reshape(-1, 1))
print('Accuracy Score: ', regressor.score(X_test.reshape(-1, 1), Y_test))

""" A figure to see it all
"""
plt.figure()
plt.scatter(X_train, Y_train, color='blue')
plt.scatter(X_test, Y_test, color='green')
plt.scatter(X_test, Y_pred, color='red')
plt.show()

""" Print the tree
"""
if printTree:
    dot_data = StringIO()
    tree.export_graphviz(regressor, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names='x',
                    class_names=['0', '1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png("fullTree.png")