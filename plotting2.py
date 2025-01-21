import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
from scipy.stats import skew

df=pd.read_csv('processed_data.csv')

for i,column in enumerate(df.columns):
    plt.figure(figsize=(20,15))
    if column=='year' or column=='seats':
        sns.countplot(x=df[column])
    elif pd.api.types.is_numeric_dtype(df[column]):
        sns.histplot(x=df[column],kde=True)
    else:
        sns.countplot(x=df[column])
    plt.title(column)
    plt.show()