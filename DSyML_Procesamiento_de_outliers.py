import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../DATA/Ames_Housing_Data.csv')
df.head()
df.corr()
df.corr()['SalePrice'].sort_values()
sns.scatterplot(data=df,x='Overall Qual',y='SalePrice')
sns.scatterplot(data=df,x='Overall Qual',y='SalePrice',hue='Overall Cond')
sns.scatterplot(data=df,x='Gr Liv Area',y='SalePrice')
df[(df['Overall Qual']>8) & (df['SalePrice']<200000)]
df[(df['Gr Liv Area']>4000) & (df['SalePrice']<200000)]
drop_ind = df[(df['Gr Liv Area']>4000) & (df['SalePrice']<400000)].index
df = df.drop(drop_ind,axis=0)
sns.scatterplot(data=df,x='Gr Liv Area',y='SalePrice')
df.to_csv('../DATA/Ames_sin_outliers.csv')