import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
from scipy.stats import skew

df=pd.read_csv('processed_data.csv')


# در اینجا نمودار پراکندگی را برای هریک از ستون های دیتاست رسم میکنیم
n_columns=3
n_rows=ceil(len(df.columns)/n_columns)
fig,axes=plt.subplots(nrows=n_rows,ncols=n_columns,figsize=(n_columns*4,n_rows*2))

for i,column in enumerate(df.columns):
    row, col = divmod(i, n_columns)
    ax = axes[row, col]
    if pd.api.types.is_numeric_dtype(df[column]):
        sns.histplot(x=df[column], ax=ax)
        # مقدار چولگی را برای ان ستون بدست می اوریم
        print(f'Skewness of {column}: {skew(df[column])}')
    else:
        sns.countplot(x=df[column], ax=ax)
    # ax.set_title(column)

plt.tight_layout()
plt.show()