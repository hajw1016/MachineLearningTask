import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
from sklearn.metrics import r2_score
from DataIndex import *

###################
## Input Data
###################
filename = 'C:/Users/User/Downloads/imports-85.data'
with open(filename) as data:
    lines = data.readlines()

numbers = []
for line in lines:
    numbers.append(line.rstrip().split(','))

data = pd.DataFrame(numbers, columns=columns_name)

###########################
###Data Preprocessing
###########################
del data['Symboling']

data[continuous_columns] = data[continuous_columns].apply(pd.to_numeric, errors='coerce')
data = data.dropna()

data_numerics_only = data.select_dtypes(include=np.number)
data_string_only = data.select_dtypes(include=object)

data_one_hot = pd.get_dummies(data_string_only)
data_preprocessed = pd.concat([data_numerics_only, data_one_hot], axis=1,join="inner")

# for column in data_numerics_only.columns:
#     plt.figure()
#     data_preprocessed.boxplot([column])
#
# scatter_matrix(data_preprocessed)
# cormat = data_preprocessed.corr()
# round(cormat, 2)

# 데이터 시각화 분석 결과 Compression Ratio 제외 결정
del data_preprocessed['Compression_Ratio']


############################
## Regression (K-Fold)
############################
y = pd.DataFrame(data_preprocessed["Normalized_Losses"])
x = pd.DataFrame(data_preprocessed.iloc[:,1:])

def build_model():
    model = Sequential()
    model.add(Dense(32, input_dim=len(x.columns), activation='relu', kernel_initializer='uniform'))
    model.add(Dense(16, activation='relu', kernel_initializer='uniform'))
    model.add(Dense(1, kernel_initializer='uniform'))
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01))
    return model

LinearModel = LinearRegression()
scores = []
nn_scores = []
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for i, (train, test) in enumerate(kfold.split(x, y)):
    LinearModel.fit(x.iloc[train,:], y.iloc[train,:])
    score = LinearModel.score(x.iloc[test,:], y.iloc[test,:])
    scores.append(score)
    model = build_model()
    model.fit(x.iloc[train,:], y.iloc[train,:], epochs=1000, batch_size=10)
    nn_scores.append(r2_score(y.iloc[test,:],model.predict(x.iloc[test,:])))

print("Linear Regression R-Square: \n", scores)
print("Neural Network R-Squaure : \n", nn_scores)
