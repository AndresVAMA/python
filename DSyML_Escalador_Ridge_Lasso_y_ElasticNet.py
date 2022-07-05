import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Advertising.csv")
X = df.drop('sales',axis=1)
y = df['sales']
###Poly###
from sklearn.preprocessing import PolynomialFeatures
polynomial_converter = PolynomialFeatures(degree=3,include_bias=False)
poly_features = polynomial_converter.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.33, random_state=101)
###Escalador###
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
###Ridge###
from sklearn.linear_model import Ridge
ridge_model = Ridge(alpha=10)
ridge_model.fit(X_train,y_train)
test_predictions = ridge_model.predict(X_test)

from sklearn.metrics import mean_absolute_error,mean_squared_error
MAE = mean_absolute_error(y_test,test_predictions)
MAE
RMSE = np.sqrt(mean_squared_error(y_test,test_predictions))
RMSE
###RidgeCV###
from sklearn.linear_model import RidgeCV
ridge_cv_model = RidgeCV(alphas=(0.1, 1.0, 10.0))
ridge_cv_model.fit(X_train,y_train)
ridge_cv_model.alpha_

from sklearn.metrics import SCORERS
ridge_cv_model = RidgeCV(alphas=(0.1, 1.0, 10.0),scoring='neg_mean_absolute_percentage_error')
ridge_cv_model.fit(X_train,y_train)
ridge_cv_model.alpha_
test_predictions = ridge_cv_model.predict(X_test)
MAE = mean_absolute_error(y_test,test_predictions)
RMSE = np.sqrt(mean_squared_error(y_test,test_predictions))
MAE
RMSE
ridge_cv_model.best_score_
###LassoCV###
from sklearn.linear_model import LassoCV
lasso_cv_model = LassoCV(eps=0.001,n_alphas=100,cv=5,max_iter=1000000)
lasso_cv_model.fit(X_train,y_train)
lasso_cv_model.alpha_
test_predictions = lasso_cv_model.predict(X_test)
MAE = mean_absolute_error(y_test,test_predictions)
RMSE = np.sqrt(mean_squared_error(y_test,test_predictions))
MAE
RMSE
###ElasticNetCV###
from sklearn.linear_model import ElasticNetCV
elastic_model = ElasticNetCV(l1_ratio=[.1, .5, .7,.9, .95, .99, 1],eps=0.001,n_alphas=100,max_iter=1000000)
elastic_model.fit(X_train,y_train)
elastic_model.l1_ratio_
elastic_model.alpha_
test_predictions = elastic_model.predict(X_test)
MAE = mean_absolute_error(y_test,test_predictions)
RMSE = np.sqrt(mean_squared_error(y_test,test_predictions))
MAE
RMSE