import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore


df=pd.read_csv('used_cars.csv')
# print(df.head())
# print(df.shape)
# print(df.columns)
# print(df.dtypes)
# print(df.isnull().sum())
# print(df.info())
# print(df.duplicated())
# print(df.describe())
# print(df.nunique())

# برخی از داده ها به صورت رشته ای هستند که باید قسمت عددی ان رشته را جدا کنیم
def extract_numeric_part(str1):
    list1=list(str1)
    list2=list(str1)
    for item in list1:
        if (item.isnumeric() or item=='.'):
            pass
        else:
            list2.remove(item)
    return ''.join(list2)

# تابعی برای حذف داده های پرت
def remove_outlier(df,columns):
    for column in columns:
        df['zscore']=zscore(df[column])
        df=df[(df['zscore']>-3)&(df['zscore']<3)]
        df.reset_index(drop=True,inplace=True)
        df=df.drop(['zscore'],axis=1)  
    return df      

# رکورد های تکراری را حذف میکنیم
df.drop_duplicates(inplace=True)

# جدا کردن قسمت عددی رشته و تبدیل نوع داده در ستون ها به نوع عددی
df['mileage']=df['mileage'].apply(lambda x:str(x))
df['mileage']=df['mileage'].apply(lambda x:extract_numeric_part(x))
df['mileage']=pd.to_numeric(df['mileage'])
df['mileage']=df['mileage'].fillna(round(df['mileage'].mean()))


df['engine']=df['engine'].apply(lambda x:str(x))
df['engine']=df['engine'].apply(lambda x:extract_numeric_part(x))
df['engine']=pd.to_numeric(df['engine'])
df['engine']=df['engine'].fillna(round(df['engine'].mean()))

df['max_power']=df['max_power'].apply(lambda x:str(x))
df['max_power']=df['max_power'].apply(lambda x:extract_numeric_part(x))
df['max_power']=pd.to_numeric(df['max_power'])
df['max_power']=df['max_power'].fillna(round(df['max_power'].mean()))

# حذف داده های پرت
df=remove_outlier(df,['selling_price','km_driven','mileage','engine','max_power'])

# پرکردن مقادیر گم شده
df['seats']=df['seats'].fillna(df['seats'].median())

# df['year']=df['year'].fillna(df['year'].median())

df.drop(['name','torque'],axis=1,inplace=True)

# df.to_csv('processed_data.csv',index=False)


# لیبل گذاری برای داده های کتگوریکال
oe=ce.OrdinalEncoder()
df['owner']=oe.fit_transform(df['owner'])

le=LabelEncoder()
df['fuel']=le.fit_transform(df['fuel'])
df['transmission']=le.fit_transform(df['transmission'])
df['seller_type']=le.fit_transform(df['seller_type'])

sc=StandardScaler()
data=sc.fit_transform(df)

df2=pd.DataFrame(data=data,columns=df.columns)

# ذخیره ی داده های پردازش شده
X=df2.drop(['selling_price'],axis=1)
y=df2[['selling_price']]

X.to_csv('X_data.csv',index=False)
y.to_csv('y_data.csv',index=False)

print(df2.corr())