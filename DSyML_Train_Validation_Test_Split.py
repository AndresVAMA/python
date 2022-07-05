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
##Clean and adjust data as necessary for X and y##
X = df.drop('sales',axis=1)
y = df['sales']
##Split Data in Train/Validation/Test for both X and y##
X_train, X_other, y_train, y_other = train_test_split(X, y, test_size=0.33, random_state=101)
#Si pongo test_size=0.5 entonces es el 50% del 30% que agarramos
X_eval, X_test, y_eval, y_test = train_test_split(X_other,y_other,test_size=0.5,random_state=101)
##Fit/Train Scaler on Training X Data##
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
##Scale X Eval Data##
X_eval = scaler.transform(X_eval)
##Create Model##
model1 = Ridge(alpha=100)
##Fit/Train Model on X Train Data##
model1.fit(X_train,y_train)
##Evaluate Model on X Evaluation Data (by creating predictions and comparing to Y_eval)##
y_eval_pred = model1.predict(X_eval)
mean_squared_error(y_eval,y_eval_pred)
##Adjust Parameters as Necessary and repeat steps 5 and 6##
model2 = Ridge(alpha=1)
model2.fit(X_train,y_train)
y_eval_pred2 = model2.predict(X_eval)
mean_squared_error(y_eval,y_eval_pred2)
##Get final metrics on Test set (not allowed to go back and adjust after this!)##
y_final_test_pred = model2.predict(X_test)
mean_squared_error(y_test,y_final_test_pred)