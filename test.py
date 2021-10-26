import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.model_selection

df = pd.read_csv('/Users/benoitginies/PycharmProjects/Projet_apprentissage_automatique/Projet_Apprentisage_Automatiqur/drinking_water_potability.csv')

data = df[['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon',
           'Trihalomethanes', 'Turbidity']]
labels = df['Potability']

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, labels, test_size=0.30, random_state=56)

print(y_train)
print(pd.concat([X_train, y_train], axis = 1, join='inner'))
print(X_train)

#print(data.isna().sum() / len(df))

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
