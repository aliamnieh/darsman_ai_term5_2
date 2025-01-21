import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error

X=pd.read_csv('X_data.csv')
# دو تا از ستون های تاثیر گذار را برای مدل چند متغیره انتخاب کردیم
X=X[['max_power','engine']]
y=pd.read_csv('y_data.csv')

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(X_train,y_train)
y_train_pred=model.predict(X_train)
y_pred=model.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(f'metrics for test data ->  mse = {mse}   r2 = {r2}')


# برای رسم نمودار از شکل سه بعدی استفاده کردیم
# محور ایکس و وای مقدار دو ستون مورد نظر را نشان میدهند
# محور زد مقدار قیمت فروش را نشان میدهد
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
surf = ax.plot_surface(X['max_power'], X['engine'], y_pred, cmap='viridis')
# ax.scatter(X['max_power'], X['engine'], y_test, color='red', s=50, label='Points')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
ax.set_xlabel('max_power')
ax.set_ylabel('engine')
ax.set_zlabel('selling_price')
plt.show()





