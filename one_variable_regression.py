import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

X=pd.read_csv('X_data.csv')
y=pd.read_csv('y_data.csv')

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# model=LinearRegression()
# model.fit(X_train[['max_power']],y_train)
# y_train_pred=model.predict(X_train[['max_power']])
# mse=mean_squared_error(y_train,y_train_pred)
# r2=r2_score(y_train,y_train_pred)

# print(mse)
# print(r2)




# تمام ستون ها را بررسی میکنیم تا به عنوان ورودی مدل انتخاب شوند
# در هر مرحله امتیاز را برای ان مدل بدست می اوریم
# سپس نمودار متناسب با ان ستون را رسم میکنیم
for i,column in enumerate(X.columns):
    model=LinearRegression()
    model.fit(X_train[[column]],y_train)
    y_pred=model.predict(X_test[[column]])

    mse=mean_squared_error(y_test,y_pred)
    r2=r2_score(y_test,y_pred)
    print(f'column = {column}  mse = {mse}  r2 = {r2}')
    print('********************************************')

    plt.scatter(X_test[column],y_test,c='blue')
    plt.plot(X_test[column],y_pred,c='green')
    plt.title(f'Linear Regression for column : {column}')
    plt.xlabel(column)
    plt.ylabel('selling_price')
    plt.show()