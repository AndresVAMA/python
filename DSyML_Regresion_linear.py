import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Advertising.csv")
df.head()

sns.pairplot(df)

X = df.drop('sales',axis=1)
y = df['sales']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
model.predict(X_test)
test_predictions = model.predict(X_test)

from sklearn.metrics import mean_absolute_error,mean_squared_error
mean_absolute_error(y_test,test_predictions)
np.sqrt(mean_squared_error(y_test,test_predictions))
test_residuals = y_test - test_predictions
test_residuals

sns.scatterplot(x=y_test,y=test_residuals)
plt.axhline(y=0,color='r',ls='--')
sns.displot(test_residuals,bins=25,kde=True)

import scipy as sp
# Create a figure and axis to plot on
fig, ax = plt.subplots(figsize=(6,8),dpi=100)
# probplot returns the raw values if needed
# we just want to see the plot, so we assign these values to _
_ = sp.stats.probplot(test_residuals,plot=ax)

final_model = LinearRegression()
final_model.fit(X,y)
final_model.coef_
y_hat = final_model.predict(X)

fig,axes = plt.subplots(nrows=1,ncols=3,figsize=(16,6))

axes[0].plot(df['TV'],df['sales'],'o')
axes[0].plot(df['TV'],y_hat,'o',color='red')
axes[0].set_ylabel("Sales")
axes[0].set_title("TV Spend")

axes[1].plot(df['radio'],df['sales'],'o')
axes[1].plot(df['radio'],y_hat,'o',color='red')
axes[1].set_title("Radio Spend")
axes[1].set_ylabel("Sales")

axes[2].plot(df['newspaper'],df['sales'],'o')
axes[2].plot(df['radio'],y_hat,'o',color='red')
axes[2].set_title("Newspaper Spend");
axes[2].set_ylabel("Sales")
plt.tight_layout();

from joblib import dump,load
dump(final_model,'final_sales_model.joblib')