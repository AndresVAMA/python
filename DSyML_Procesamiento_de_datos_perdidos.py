import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../DATA/Ames_sin_outliers.csv')
df.info()
df.head()
df = df.drop('PID',axis=1)
len(df.columns)
100 * df.isnull().sum()/ len(df)

def percent_missing(df):
    percent_nan = 100 * df.isnull().sum()/ len(df)
    percent_nan = percent_nan[percent_nan >0].sort_values()
    
    return percent_nan

percent_nan = percent_missing(df)
percent_nan

plt.figure(figsize=(8,4),dpi=200)
sns.barplot(x=percent_nan.index,y=percent_nan)
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(8,4),dpi=200)
sns.barplot(x=percent_nan.index,y=percent_nan)
plt.xticks(rotation=90)
plt.ylim(0,1)
plt.show()

percent_nan[percent_nan <1]

df[df['Electrical'].isnull()]

df[df['Electrical'].isnull()]['Garage Area']

df[df['Bsmt Half Bath'].isnull()]

df = df.dropna(axis=0,subset=['Electrical','Garage Cars'])

percent_nan = percent_missing(df)
percent_nan[percent_nan<1]

plt.figure(figsize=(8,4),dpi=200)
sns.barplot(x=percent_nan.index,y=percent_nan)
plt.xticks(rotation=90)
plt.ylim(0,1)
plt.show()

df[df['Bsmt Half Bath'].isnull()]

df[df['Bsmt Full Bath'].isnull()]

df[df['Bsmt Unf SF'].isnull()]

#BSMT numeric colums -->fill Na 0
bsmt_num_cols = ['BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF','Total Bsmt SF', 'Bsmt Full Bath', 'Bsmt Half Bath']
df[bsmt_num_cols] = df[bsmt_num_cols].fillna(0)



#BSMT string columns
bsmt_str_cols =  ['Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2']
df[bsmt_str_cols] = df[bsmt_str_cols].fillna('None')

df[df['Bsmt Full Bath'].isnull()]

percent_nan = percent_missing(df)

plt.figure(figsize=(8,4),dpi=200)
sns.barplot(x=percent_nan.index,y=percent_nan)
plt.xticks(rotation=90)
plt.ylim(0,1)
plt.show()

df['Mas Vnr Type'] = df['Mas Vnr Type'].fillna("None")

df['Mas Vnr Area'] = df['Mas Vnr Area'].fillna(0)

percent_nan = percent_missing(df)

plt.figure(figsize=(8,4),dpi=200)
sns.barplot(x=percent_nan.index,y=percent_nan)
plt.xticks(rotation=90)
plt.show()

gar_str_cols = ['Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond']
df[gar_str_cols] = df[gar_str_cols].fillna('None')

df['Garage Yr Blt'] = df['Garage Yr Blt'].fillna(0)

percent_nan = percent_missing(df)

plt.figure(figsize=(8,4),dpi=200)
sns.barplot(x=percent_nan.index,y=percent_nan)
plt.xticks(rotation=90)
plt.show()

df = df.drop(['Pool QC','Misc Feature','Alley','Fence'],axis=1)

percent_nan = percent_missing(df)

plt.figure(figsize=(8,4),dpi=200)
sns.barplot(x=percent_nan.index,y=percent_nan)
plt.xticks(rotation=90)
plt.show()

df['Fireplace Qu'].value_counts()

df['Fireplace Qu'] = df['Fireplace Qu'].fillna('None')

df['Lot Frontage']

plt.figure(figsize=(8,12))
sns.boxplot(x='Lot Frontage',y='Neighborhood',data=df,orient='h')

df.groupby('Neighborhood')['Lot Frontage']

df.groupby('Neighborhood')['Lot Frontage'].mean()

df.groupby('Neighborhood')['Lot Frontage'].transform(lambda value: value.fillna(value.mean()))

df['Lot Frontage'] = df.groupby('Neighborhood')['Lot Frontage'].transform(lambda value: value.fillna(value.mean()))

percent_nan = percent_missing(df)

plt.figure(figsize=(8,4),dpi=200)
sns.barplot(x=percent_nan.index,y=percent_nan)
plt.xticks(rotation=90)
plt.show()

