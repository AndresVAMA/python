import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../DATA/Advertising.csv")
df.head()

###Train-Test split###
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
##Clean and adjust data as necessary for X and y##
X = df.drop('sales',axis=1)
y = df['sales']
##Split Data in Train/Test for both X and y##
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
##Fit/Train Scaler on Training X Data##
scaler = StandardScaler()
scaler.fit(X_train)
##Scale X Test Data##
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
##Create Model##
model = Ridge(alpha=100)
##Fit/Train Model on X Train Data##
model.fit(X_train,y_train)
##Evaluate Model on X Test Data (by creating predictions and comparing to Y_test)##
y_pred = model.predict(X_test)
##Adjust Parameters as Necessary and repeat steps 5 and 6##
mean_squared_error(y_test,y_pred)
##Adjust Parameters as Necessary and repeat steps 5 and 6##
model2 = Ridge(alpha=1)
model2.fit(X_train,y_train)
y_pred2 = model2.predict(X_test)
mean_squared_error(y_test,y_pred2)