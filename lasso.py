import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

X=pd.read_csv('X_data.csv')
y=pd.read_csv('y_data.csv')

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=Lasso()
params={
    'alpha':np.arange(0.01,1,0.01)
}
grid_search=GridSearchCV(estimator=model,param_grid=params,scoring='r2',cv=5,verbose=3)
grid_search.fit(X_train,y_train)
print(f'best parameters : {grid_search.best_params_}')
print(f'best score : {grid_search.best_score_}')
best_model=grid_search.best_estimator_
print('validation')
print(f'mse = {mean_squared_error(y_test,best_model.predict(X_test))}')
print(f'r2 = {r2_score(y_test,best_model.predict(X_test))}')

# در اینجا نمودار متناسب را میکشیم
# چون نمودار ما دو بعدی است از مدل تک متغیره استفاده میکنیم
X=X[['max_power']]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=Lasso()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
plt.scatter(X_test,y_test,color='green')
plt.plot(X_test,y_pred,color='blue')
plt.xlabel('max_power')
plt.ylabel('selling_price')
plt.title('regression')
plt.show()