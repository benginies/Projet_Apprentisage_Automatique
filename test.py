import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.model_selection
import sklearn.decomposition
import sklearn.impute
import sklearn.preprocessing
import sklearn.ensemble
import sklearn.utils
import sklearn.svm
import sklearn.naive_bayes
import sklearn.linear_model

dir = os.getcwd()

df = pd.read_csv(dir + '/Data/drinking_water_potability.csv')

data = df[['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon',
           'Trihalomethanes', 'Turbidity']]
labels = df['Potability']

data, labels = sklearn.utils.shuffle(data, labels)

data_pot = df.loc[df['Potability'] == 1]
data_npot = df.loc[df['Potability'] == 0]

print(len(data_pot) / len(df))
print(len(data_npot) / len(df))

#X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, labels, test_size=0.30, random_state=56)

corr_matrix = df.corr()

#Normalizing data with mean and std

normalizer = sklearn.preprocessing.StandardScaler()
data = pd.DataFrame(normalizer.fit_transform(data), columns=data.columns)

#corr_matrix.to_excel(dir + '/features_correlation.xlsx')

print(data.isna().sum() / len(df))
print(len(df))

#Predict missing data with a regression

KNNImpute = sklearn.impute.KNNImputer(n_neighbors = 15)
KNNdata = pd.DataFrame(KNNImpute.fit_transform(data), columns=data.columns)

MeanImpute = sklearn.impute.SimpleImputer(strategy = 'mean')
Meandata = pd.DataFrame(MeanImpute.fit_transform(data), columns=data.columns)

potMeanImpute = sklearn.impute.SimpleImputer(strategy = 'mean')
npotMeanImpute = sklearn.impute.SimpleImputer(strategy = 'mean')
potMeandata = pd.DataFrame(potMeanImpute.fit_transform(data_pot), columns=data_pot.columns)
npotMeandata = pd.DataFrame(npotMeanImpute.fit_transform(data_npot), columns=data_npot.columns)
SepMeandata = pd.concat([potMeandata, npotMeandata], axis  = 0)
SepMeanlabels = SepMeandata['Potability'].copy()
SepMeandata = SepMeandata.drop(['Potability'], axis = 1)
SepMeandata, SepMeanlabels = sklearn.utils.shuffle(SepMeandata, SepMeanlabels)


#Predict missing data with mean

#Compute PCA and explore feature importance

# PCA = sklearn.decomposition.PCA(n_components = 9)
# PCA.fit(data)
#
# print(PCA.explained_variance_ratio_)

#Various models

#RF

#RF grid search (with cv)

#RFparameters = {'n_estimators' : [10,100,500], 'max_depth' : [7, 10, 12], 'min_samples_leaf' : [1, 10, 50]}
RFparameters = {'n_estimators' : [500, 750, 1000], 'max_depth' : [8, 10, 11], 'min_samples_leaf' : [1]}
#parameters = {'n_estimators' : [10, 100], 'max_depth' : [5], 'min_samples_leaf' : [1]}


# KNNRF = sklearn.ensemble.RandomForestClassifier()
# GSKNNRF = sklearn.model_selection.GridSearchCV(KNNRF, RFparameters, verbose = 2)
# GSKNNRF.fit(KNNdata, labels)
# KNNRFscores = GSKNNRF.best_score_
# print("KNN imputer, RF",KNNRFscores)
#
# MeanRF = sklearn.ensemble.RandomForestClassifier()
# GSMeanRF = sklearn.model_selection.GridSearchCV(MeanRF, RFparameters, verbose = 2)
# GSMeanRF.fit(Meandata, labels)
# MeansRFcores = GSMeanRF.best_score_
# print("Mean imputer, RF",MeansRFcores)
#
# SepMeanRF = sklearn.ensemble.RandomForestClassifier()
# GSSepMeanRF = sklearn.model_selection.GridSearchCV(SepMeanRF, RFparameters, verbose = 2)
# GSSepMeanRF.fit(SepMeandata, SepMeanlabels)
# SepMeanRFcores = GSSepMeanRF.best_score_
# print("Sep Mean imputer, RF",SepMeanRFcores)
# print("Best params",GSSepMeanRF.best_params_)

# KNNRF = sklearn.ensemble.RandomForestClassifier(n_estimators = 100)
# KNNscores = sklearn. model_selection.cross_val_score(KNNRF, KNNdata, labels , cv=5)
# print("KNN imputer, RF",KNNscores)
#
# MeanRF = sklearn.ensemble.RandomForestClassifier(n_estimators = 100)
# Meanscores = sklearn. model_selection.cross_val_score(MeanRF, Meandata, labels , cv=5)
# print("Mean imputer, RF",Meanscores)
#
# SepMeanRF = sklearn.ensemble.RandomForestClassifier(n_estimators = 100)
# SepMeanscores = sklearn. model_selection.cross_val_score(SepMeanRF, SepMeandata, SepMeanlabels , cv=5)
# print("SepMean imputer, RF",SepMeanscores)

#SVC

SVCparameters = {'C' : [0.1, 1, 10], 'kernel' : ['linear', 'poly', 'sigmoid']}

# KNNSVC = sklearn.svm.SVC()
# GSKNNSVC = sklearn.model_selection.GridSearchCV(KNNSVC, SVCparameters, verbose = 2)
# GSKNNSVC.fit(KNNdata, labels)
# KNNSVCscores = GSKNNSVC.best_score_
# print("KNN imputer, SVC", KNNSVCscores)
#
# MeanSVC = sklearn.svm.SVC()
# GSMeanSVC = sklearn.model_selection.GridSearchCV(MeanSVC, SVCparameters, verbose = 2)
# GSMeanSVC.fit(Meandata, labels)
# MeanSVCscores = GSMeanSVC.best_score_
# print("Mean imputer, SVC", MeanSVCscores)
#
# SepMeanSVC = sklearn.svm.SVC()
# GSSepMeanSVC = sklearn.model_selection.GridSearchCV(SepMeanSVC, SVCparameters, verbose = 2)
# GSSepMeanSVC.fit(SepMeandata, SepMeanlabels)
# SepMeanSVCscores = GSSepMeanSVC.best_score_
# print("Sep Mean imputer, SVC", SepMeanSVCscores)

#Adaboost

ABparameters = {'n_estimators' : [125, 150, 175], 'learning_rate' : [0.4, 0.5, 0.6]}
#
# KNNAB = sklearn.ensemble.AdaBoostClassifier()
# GSKNNAB = sklearn.model_selection.GridSearchCV(KNNAB, ABparameters, verbose = 2)
# GSKNNAB.fit(KNNdata, labels)
# KNNABscores = GSKNNAB.best_score_
# print("KNN imputer, AB", KNNABscores)
#
# MeanAB = sklearn.ensemble.AdaBoostClassifier()
# GSMeanAB = sklearn.model_selection.GridSearchCV(MeanAB, ABparameters, verbose = 2)
# GSMeanAB.fit(Meandata, labels)
# MeanABscores = GSMeanAB.best_score_
# print("Mean imputer, AB", MeanABscores)
#
# SepMeanAB = sklearn.ensemble.AdaBoostClassifier()
# GSSepMeanAB = sklearn.model_selection.GridSearchCV(SepMeanAB, ABparameters, verbose = 2)
# GSSepMeanAB.fit(SepMeandata, SepMeanlabels)
# SepMeanABscores = GSSepMeanAB.best_score_
# print("Sep Mean imputer, AB", SepMeanABscores)
# print("Best params", GSSepMeanAB.best_params_)

#Gaussian Naive Bayes

# KNNGNB = sklearn.naive_bayes.GaussianNB()
# KNNGNBscores = sklearn. model_selection.cross_val_score(KNNGNB, KNNdata, labels , cv=5)
# print("KNN imputer, GNB",KNNGNBscores.mean())
#
# MeanGNB = sklearn.naive_bayes.GaussianNB()
# MeanGNBscores = sklearn. model_selection.cross_val_score(MeanGNB, Meandata, labels , cv=5)
# print("Mean imputer, GNB",MeanGNBscores.mean())
#
# SepMeanGNB = sklearn.naive_bayes.GaussianNB()
# SepMeanGNBscores = sklearn. model_selection.cross_val_score(SepMeanGNB, SepMeandata, SepMeanlabels , cv=5)
# print("SepMean imputer, GNB",SepMeanGNBscores.mean())

#Stacking with best examples

estimators = [
    ('RF', sklearn.ensemble.RandomForestClassifier(n_estimators=750, max_depth=11, min_samples_leaf=1)),
    ('AB', sklearn.ensemble.AdaBoostClassifier(learning_rate=0.5, n_estimators=125))
]

stack = sklearn.ensemble.StackingClassifier(estimators=estimators, final_estimator=sklearn.linear_model.LogisticRegression())
SepMeanStackscores = sklearn. model_selection.cross_val_score(stack, SepMeandata, SepMeanlabels , cv=5, verbose = 1)
print("SepMean imputer, RF",SepMeanStackscores.mean())

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
