import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
from sklearn import svm
import xgboost
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import pickle

#Importing dataset
data=pd.read_csv("traffic volume.csv")

data=pd.read_csv("traffic volume.csv")
data.head()
data.describe()
data.info()
data.isnull().sum()

from collections import Counter

print(Counter(data['weather']))

#Handling missing values
data.fillna({
    'temp': data['temp'].mean(),
    'rain': data['rain'].mean(),
    'snow': data['snow'].mean(),
    'weather': 'Clouds'
}, inplace=True)
data['holiday'] = data['holiday'].fillna('None')

# --- Split Date & Time ---
data[['day', 'month', 'year']] = data['date'].str.split('-', expand=True).astype(int)
data[['hours', 'minutes', 'seconds']] = data['Time'].str.split(':', expand=True).astype(int)
data.drop(columns=['date', 'Time'], inplace=True)

# --- Encode Categorical Variables ---
#---We have two columns with categorical variables holiday and weather---
#--Ecoding those with the help of label encoder---

def encode_and_save(column, name):
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    pickle.dump(le, open(f'le_{name}.pkl', 'wb'))
    return data[column]

data['holiday'] = encode_and_save('holiday', 'holiday')
data['weather'] = encode_and_save('weather', 'weather')

columns_for_correlation = ['temp', 'rain', 'snow', 'traffic_volume'] 
correlation_matrix = data[columns_for_correlation].corr()
print(correlation_matrix)

#Data visualization
sns.pairplot(data)
data.boxplot()

#splitting the features into dependent and independent variables
y = data['traffic_volume']
X = data.drop('traffic_volume', axis=1)

# --- Feature Scaling ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pickle.dump(scaler, open('scaler.pkl', 'wb'))

#Splitting the data into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 42)

lin_reg=linear_model.LinearRegression()
lin_reg.fit(x_train,y_train)
Dtree=tree.DecisionTreeRegressor()
Dtree.fit(x_train,y_train)
Rforest=ensemble.RandomForestRegressor()
Rforest.fit(x_train,y_train)
svr=svm.SVR()
svr.fit(x_train,y_train)
xgb=xgboost.XGBRegressor()
xgb.fit(x_train,y_train)

#--Checking r2_score for all the models
p1=lin_reg.predict(x_train)
p2=Dtree.predict(x_train)
p3=Rforest.predict(x_train)
p4=svr.predict(x_train)
p5=xgb.predict(x_train)

from sklearn import metrics
print(metrics.r2_score(p1,y_train))
print(metrics.r2_score(p2,y_train))
print(metrics.r2_score(p3,y_train))
print(metrics.r2_score(p4,y_train))
print(metrics.r2_score(p5,y_train))

a1=lin_reg.predict(x_test)
a2=Dtree.predict(x_test)
a3=Rforest.predict(x_test)
a4=svr.predict(x_test)
a5=xgb.predict(x_test)

print(metrics.r2_score(a1,y_test))
print(metrics.r2_score(a2,y_test))
print(metrics.r2_score(a3,y_test))
print(metrics.r2_score(a4,y_test))
print(metrics.r2_score(a5,y_test))

#--By checking the performance scores of each model we have got to a conclusion that Random forest regressor is the best fit for building the model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Predictions from the Random Forest model
predictions = a3

# Evaluation Metrics
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

# Display results
print("📊 Evaluation Metrics for Random Forest:")
print(f"MAE  (Mean Absolute Error):      {mae:.2f}")
print(f"MSE  (Mean Squared Error):       {mse:.2f}")
print(f"RMSE (Root Mean Squared Error):  {rmse:.2f}")
print(f"R2 Score:                        {r2:.4f}")

from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, None],
    'min_samples_split': [2, 5]
}

rf = ensemble.RandomForestRegressor(random_state=42)

rf_search = RandomizedSearchCV(rf, param_distributions=param_grid,
                               n_iter=6, cv=3, n_jobs=-1, verbose=1)

rf_search.fit(x_train, y_train)
print("Best params:", rf_search.best_params_)


pickle.dump(Rforest,open('model.pkl','wb'))

