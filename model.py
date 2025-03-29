# Data Manipulation
import pandas as pd
import numpy as np
# Data Visualisation
import seaborn as sns
import matplotlib.pyplot as plt

# Loading dataset for the provided url
url = 'https://raw.githubusercontent.com/Muskansingh-22/Ml_Model/refs/heads/main/Concrete_Data.csv'

df = pd.read_csv(url)

# Split the Dataset into feature(X) and Target(Y)
x = df.drop(columns = 'Concrete compressive strength(MPa, megapascals) ',axis = 1)
y = df['Concrete compressive strength(MPa, megapascals) ']
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state= 42)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)  # Seen data
x_test = scaler.transform(x_test)        # Unseen data

# Model Building
from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor()
RF.fit(x_train,y_train)
y_pred_RF = RF.predict(x_test)
from sklearn.metrics import r2_score
r2_score_RF= r2_score(y_test,y_pred_RF)
print(f'The Model R2 Score is {r2_score_RF*100} %')