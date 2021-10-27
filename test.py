import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.model_selection
import sklearn.decomposition
import sklearn.impute
import sklearn.preprocessing

dir = os.getcwd()

df = pd.read_csv(dir + '/drinking_water_potability.csv')

data = df[['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon',
           'Trihalomethanes', 'Turbidity']]
labels = df['Potability']

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, labels, test_size=0.30, random_state=56)

corr_matrix = df.corr()

#Normalizing data with mean and std

normalizer = sklearn.preprocessing.StandardScaler()
data = pd.DataFrame(normalizer.fit_transform(data), columns=data.columns)

#corr_matrix.to_excel(dir + '/features_correlation.xlsx')

print(data.isna().sum() / len(df))
print(len(df))

#Predict missing data with a regression

KNNImput = sklearn.impute.KNNImputer(n_neighbors = 15)
data = pd.DataFrame(KNNImput.fit_transform(data), columns=data.columns)

#Compute PCA and explore feature importance

PCA = sklearn.decomposition.PCA(n_components = 9)
PCA.fit(data)

print(PCA.explained_variance_ratio_)

#Various models


###### Rome

#df.fillna(df.mean(), inplace=True)
# df0 = df.loc[df['Potability'] == 0, :]
# df1 = df.loc[df['Potability'] == 1, :]
# df0['ph'].fillna(value=df['ph'].mean(), inplace=True)
# df0['Sulfate'].fillna(value=df['Sulfate'].mean(), inplace=True)
# df0['Trihalomethanes'].fillna(value=df['Trihalomethanes'].mean(), inplace=True)
# df1['ph'].fillna(value=df1['ph'].mean(), inplace=True)
# df1['Sulfate'].fillna(value=df1['Sulfate'].mean(), inplace=True)
# df1['Trihalomethanes'].fillna(value=df1['Trihalomethanes'].mean(), inplace=True)
# df = pd.concat([df0, df1], axis=0)
