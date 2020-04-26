import pandas as pd
import shap
import sklearn.model_selection as ms
import xgboost as xgb

data0 = pd.read_csv('data/atmosphere/k0_1200.csv')
data1 = pd.read_csv('data/atmosphere/k1_1200.csv')
data2 = pd.read_csv('data/atmosphere/k2_1200.csv')
data3 = pd.read_csv('data/atmosphere/k3_1200.csv')

frames = [data0, data1, data2, data3]
allData = pd.concat(frames, ignore_index=True)


def BDT(X, Y):
    """ Split for training and testing
    """
    x_train, x_test, y_train, y_test = ms.train_test_split(X.values, Y, test_size=0.20, random_state=0)
    x_strain, x_valid, y_strain, y_valid = ms.train_test_split(x_train, y_train)

    """ Fit the decision tree
    """
    regressor = xgb.XGBRegressor(learning_rate=0.02, subsample=0.2, colsample_bytree=0.5, n_estimators=5000, base_score=y_strain.mean())
    regressor = regressor.fit(x_strain, y_strain, eval_set=[(x_valid,y_valid)], eval_metric="logloss", verbose=500, early_stopping_rounds=20)

    """ Predictions
    """
    y_pred = regressor.predict(x_test)
    print('Accuracy Score: ', regressor.score(x_test, y_test))

    """ shapIT
    """
    explainer = shap.TreeExplainer(regressor)
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, X[:8000], plot_type='violin')
    shap.summary_plot(shap_values, X[:8000], plot_type='bar')


""" Split into dependent and independent variables
"""
allData = allData.drop(['Phi distribution'], axis=1)
X = allData.iloc[:, :-1]
Y = allData.iloc[:, -1].values

""" Run the Forest
"""
BDT(X,Y)

""" Check for certain features
"""
# X_red1 = allData[['Phi distribution', 'CH4 clustering coefficient', 'Average clustering coefficient', 'NH3 clustering coefficient']]
# X_red2 = allData[['Phi distribution']]
# X_red3 = allData[['CH4 clustering coefficient', 'Average clustering coefficient', 'NH3 clustering coefficient']]
#
# BDT(X_red1, Y)
# BDT(X_red2, Y)
# BDT(X_red3, Y)