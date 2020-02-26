import pandas as pd
import shap

from sklearn import ensemble
import sklearn.model_selection as ms
from sklearn import metrics

data0 = pd.read_csv('data/atmosphere/k0_1200.csv')
data1 = pd.read_csv('data/atmosphere/k1_1200.csv')
data2 = pd.read_csv('data/atmosphere/k2_1200.csv')
data3 = pd.read_csv('data/atmosphere/k3_1200.csv')

frames = [data0, data1, data2, data3]
allData = pd.concat(frames, ignore_index=True)


def RandomForest(X, Y):
    """ Split for training and testing
    """
    x_train, x_test, y_train, y_test = ms.train_test_split(X.values, Y, test_size=0.25, random_state=0)

    """ Fit the decision tree
    """
    classifier = ensemble.RandomForestClassifier()
    classifier = classifier.fit(x_train, y_train)

    """ Predictions
    """
    y_pred = classifier.predict(x_test)
    print('Accuracy Score: ', metrics.accuracy_score(y_test, y_pred))

    """ shapIT
    """
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, X, plot_type='bar', class_names=["k = 0", "k = 1e06", "k = 1e08", "k = 1e10"],
                      class_inds='original')


""" Split into dependent and independent variables
"""
X = allData.iloc[:, :-1]
Y = allData.iloc[:, -1].values

""" Run the Forest
"""
RandomForest(X,Y)

""" Check for certain features
"""
X_red1 = allData[['Phi distribution', 'CH4 clustering coefficient', 'Average clustering coefficient', 'NH3 clustering coefficient']]
X_red2 = allData[['Phi distribution']]
X_red3 = allData[['CH4 clustering coefficient', 'Average clustering coefficient', 'NH3 clustering coefficient']]

RandomForest(X_red1, Y)
RandomForest(X_red2, Y)
RandomForest(X_red3, Y)

