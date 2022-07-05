import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

with open('../DATA/Ames_Housing_Feature_Description.txt') as f:
    print(f.read())

df = pd.read_csv("../DATA/Ames_NO_Missing_Data.csv")
df.head()

df['MS SubClass']= df['MS SubClass'].apply(str)

direction = pd.Series(['Up','Up','Down'])
direction
pd.get_dummies(direction)
pd.get_dummies(direction,drop_first=True)
df.select_dtypes(include='object')
my_object_df = df.select_dtypes(include='object')
my_numeric_df = df.select_dtypes(exclude='object')
df_objects_dummies = pd.get_dummies(my_object_df,drop_first=True)
df_objects_dummies

final_df = pd.concat([my_numeric_df,df_objects_dummies],axis=1)
final_df
final_df.corr()['SalePrice'].sort_values()
