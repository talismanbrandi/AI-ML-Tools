import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn import ensemble
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
import pydotplus
import math as m

""" Initialize some variables and containers
"""
nx = 1000
nx2 = int(nx / 2)
printTree = False

xa = np.zeros((nx, 2))
X_train = np.zeros(nx)
Y_train = np.zeros(nx)
X_test = np.zeros(nx2)
Y_test = np.zeros(nx2)

rng = np.random.RandomState(1)

""" Generate data for fitting
"""
for i in range(0, nx):
    X_train[i] = i * 2. * m.pi / (nx - 1.) + 0.01
    Y_train[i] = m.sin(X_train[i]) * m.cos(4. * X_train[i]) * (1. + np.random.normal(0., 0.2, 1))

for i in range(0, nx2):
    X_test[i] = i * 2. * m.pi / (nx2 - 1.) + 0.01
    Y_test[i] = m.sin(X_test[i]) * m.cos(4. * X_test[i])

""" Fit a single decision tree
"""
regressor = tree.DecisionTreeRegressor(splitter='random', min_samples_split=10)
regressor = regressor.fit(X_train.reshape(-1, 1), Y_train)

""" Fit with AdaBoost
"""
regressorA = ensemble.AdaBoostRegressor(tree.DecisionTreeRegressor(min_samples_split=10, max_depth=10), n_estimators=300,
                                        learning_rate=0.001, random_state=rng)
regressorA = regressorA.fit(X_train.reshape(-1, 1), Y_train)

""" Fit with Random Forest
"""
regressorR = ensemble.RandomForestRegressor(n_estimators=100, min_samples_split=10, random_state=rng, max_depth=10)
regressorR = regressorR.fit(X_train.reshape(-1, 1), Y_train)

""" Test the regressors
"""
Y_pred = regressor.predict(X_test.reshape(-1, 1))
print('Accuracy Score: ', regressor.score(X_test.reshape(-1, 1), Y_test))

Y_predA = regressorA.predict(X_test.reshape(-1, 1))
print('Accuracy Score: ', regressorA.score(X_test.reshape(-1, 1), Y_test))

Y_predR = regressorR.predict(X_test.reshape(-1, 1))
print('Accuracy Score: ', regressorR.score(X_test.reshape(-1, 1), Y_test))

""" A figure to see it all
"""
plt.figure()
plt.scatter(X_train, Y_train, color='blue')
plt.scatter(X_test, Y_test, color='green')
plt.scatter(X_test, Y_pred, color='red')
plt.show()

plt.figure()
plt.scatter(X_train, Y_train, color='orange')
plt.scatter(X_test, Y_test, color='green')
plt.scatter(X_test, Y_predA, color='purple')
plt.show()

plt.figure()
plt.scatter(X_train, Y_train, color='yellow')
plt.scatter(X_test, Y_test, color='green')
plt.scatter(X_test, Y_predR, color='pink')
plt.show()

""" Print the tree
"""
if printTree:
    dot_data = StringIO()
    export_graphviz(regressor, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names='x',
                    class_names=['0', '1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png("fullTree.png")
