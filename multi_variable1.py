import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error

X=pd.read_csv('X_data.csv')
y=pd.read_csv('y_data.csv')

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearRegression()
# برای مدل چند متغیره همه ی ستون ها را انتخاب کردیم
model.fit(X_train,y_train)
y_train_pred=model.predict(X_train)
mse=mean_squared_error(y_train,y_train_pred)
r2=r2_score(y_train,y_train_pred)
print(f'metrics for training data ->  mse = {mse}   r2 = {r2}')
y_pred=model.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(f'metrics for test data ->  mse = {mse}   r2 = {r2}')
