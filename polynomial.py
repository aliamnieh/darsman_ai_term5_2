import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

X=pd.read_csv('X_data.csv')
y=pd.read_csv('y_data.csv')

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# در اینجا از مدل چندجمله ای استفاده میکنیم
# در هر مرحله درجه ی چند جمله ای را تعیین میکنیم
# سپس امتیاز دهی را برای ان مدل انجام میدهیم تا اورفیتینگ را بررسی کنیم
for i in range(1,6):
    pf=PolynomialFeatures(degree=i)
    X_train_poly=pf.fit_transform(X_train)
    X_test_poly=pf.transform(X_test)
    model=LinearRegression()
    model.fit(X_train_poly,y_train)
    mse=mean_squared_error(y_train,model.predict(X_train_poly))
    r2=r2_score(y_train,model.predict(X_train_poly))
    print(f'degree = {i}')
    print(f'metrics for training data -> mse = {mse}  r2 = {r2}')
    mse=mean_squared_error(y_test,model.predict(X_test_poly))
    r2=r2_score(y_test,model.predict(X_test_poly))
    print(f'metrics for test data -> mse = {mse}  r2 = {r2}')
    print('*************************************************')