import pandas as pd
import shap
import numpy as np
from sklearn import ensemble
import sklearn.model_selection as ms

data0 = pd.read_csv('/Users/ayanpaul/Codes/Interpolators/MatrixElelment/data/ggzz_grid.dat')


def Trees(X, Y, type='BDT'):
    """ Split for training and testing
    """
    x_train, x_test, y_train, y_test = ms.train_test_split(X.values, Y, test_size=0.20, random_state=0)

    """ Fit the decision tree
    """
    if type == 'BDT': regressor = ensemble.GradientBoostingRegressor()
    elif type == 'RF': regressor = ensemble.RandomForestRegressor()
    elif type == 'ADA': regressor = ensemble.AdaBoostRegressor()
    elif type == 'EXT': regressor = ensemble.ExtraTreesRegressor()
    else:
        print('type can only be BDT, RF, ADA or EXT')
        exit(1)

    regressor = regressor.fit(x_train, y_train)

    """ Accuracy Measure
    """
    print('Accuracy Score: ', regressor.score(x_test, y_test))

    """ shapIT
    """
    if type != 'EXT':
        X_test = pd.DataFrame(x_test, columns={'pT', r'cos$\theta$'})
        explainer = shap.TreeExplainer(regressor)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test, plot_type='violin')
        shap.summary_plot(shap_values, X_test, plot_type='bar')

    return regressor, x_test, y_test

#%%

""" Split into dependent and independent variables
"""
X = data0.iloc[:, :-1]
Y = np.log(data0.iloc[:, -1].values)

""" Grow the Trees
"""
regressor, _, _ = Trees(X, Y, type='BDT')

