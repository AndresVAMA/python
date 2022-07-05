import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../DATA/Advertising.csv")
df.head()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate

## CREATE X and y
X = df.drop('sales',axis=1)
y = df['sales']

##TRAIN TEST SPLIT##
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

##SCALE DATA##
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
##Modelo##
model = Ridge(alpha=100)
##SCORING##
scores = cross_validate(model,X_train,y_train,scoring=['neg_mean_absolute_error','neg_mean_squared_error','max_error'],cv=5)

model2 = Ridge(alpha=1)

scores = cross_validate(model2,X_train,y_train,scoring=['neg_mean_absolute_error','neg_mean_squared_error','max_error'],cv=5)

#Final model
# Need to fit the model first!
model2.fit(X_train,y_train)

y_final_test_pred = model2.predict(X_test)

mean_squared_error(y_test,y_final_test_pred)