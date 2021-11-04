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
import sklearn.neural_network
import sklearn.metrics

dir = os.getcwd()


#----------------------------------------------------------------------------------------------------------------------#
#### Handling Data
#----------------------------------------------------------------------------------------------------------------------#


df = pd.read_csv(dir + '/Data/drinking_water_potability.csv')

data = df[['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon',
           'Trihalomethanes', 'Turbidity']]
labels = df['Potability']

data, labels = sklearn.utils.shuffle(data, labels)

data_pot = df.loc[df['Potability'] == 1]
data_npot = df.loc[df['Potability'] == 0]

#Normalizing data with mean and std

normalizer = sklearn.preprocessing.StandardScaler()
data = pd.DataFrame(normalizer.fit_transform(data), columns=data.columns)

#----------------------------------------------------------------------------------------------------------------------#

#Predict missing data with KNN imputer

KNNImpute = sklearn.impute.KNNImputer(n_neighbors = 15)
KNNdata = pd.DataFrame(KNNImpute.fit_transform(data), columns=data.columns)

#Predict missing data with Mean imputer

MeanImpute = sklearn.impute.SimpleImputer(strategy = 'mean')
Meandata = pd.DataFrame(MeanImpute.fit_transform(data), columns=data.columns)

#Predict missing data with sep Mean Imputer

potMeanImpute = sklearn.impute.SimpleImputer(strategy = 'mean')
npotMeanImpute = sklearn.impute.SimpleImputer(strategy = 'mean')
potMeandata = pd.DataFrame(potMeanImpute.fit_transform(data_pot), columns=data_pot.columns)
npotMeandata = pd.DataFrame(npotMeanImpute.fit_transform(data_npot), columns=data_npot.columns)
SepMeandata = pd.concat([potMeandata, npotMeandata], axis  = 0)
SepMeanlabels = SepMeandata['Potability'].copy()
SepMeandata = SepMeandata.drop(['Potability'], axis = 1)
SepMeandata, SepMeanlabels = sklearn.utils.shuffle(SepMeandata, SepMeanlabels)

#Predict missing data with sep KNN imputer

potKNNImpute = sklearn.impute.KNNImputer(n_neighbors = 50)
npotKNNImpute = sklearn.impute.KNNImputer(n_neighbors = 50)
potKNNlabels = data_pot['Potability'].copy()
npotKNNlabels = data_npot['Potability'].copy()
potKNNdata = pd.DataFrame(potKNNImpute.fit_transform(data_pot.drop(['Potability'], axis = 1)), columns=data_pot.drop(['Potability'], axis = 1).columns)
npotKNNdata = pd.DataFrame(npotKNNImpute.fit_transform(data_npot.drop(['Potability'], axis = 1)), columns=data_npot.drop(['Potability'], axis = 1).columns)
SepKNNdata = pd.concat([potKNNdata, npotKNNdata], axis  = 0)
SepKNNlabels = pd.concat([potKNNlabels, npotKNNlabels], axis = 0)
SepKNNdata, SepKNNlabels = sklearn.utils.shuffle(SepKNNdata, SepKNNlabels)

#Compute PCA and explore variance explained ratios

# PCA = sklearn.decomposition.PCA(n_components = 9)
# PCA.fit(data)
#
# print(PCA.explained_variance_ratio_)


#----------------------------------------------------------------------------------------------------------------------#
####Computing CV
#----------------------------------------------------------------------------------------------------------------------#


#RF

RFparameters = {'n_estimators' : [500, 750, 1000], 'max_depth' : [8, 10, 11], 'min_samples_leaf' : [1]}

#with KNN imputer

# KNNRF = sklearn.ensemble.RandomForestClassifier()
# GSKNNRF = sklearn.model_selection.GridSearchCV(KNNRF, RFparameters, verbose = 2)
# GSKNNRF.fit(KNNdata, labels)
# KNNRFscores = GSKNNRF.best_score_
# print("KNN imputer, RF",KNNRFscores)
# GSKNNRFpred = GSKNNRF.best_estimator_
# GSKNNRFres = GSKNNRFpred.predict(KNNdata)
# print(sklearn.metrics.confusion_matrix(labels, GSKNNRFres))
# print(sklearn.metrics.classification_report(labels, GSKNNRFres))

#with Mean imputer

# MeanRF = sklearn.ensemble.RandomForestClassifier()
# GSMeanRF = sklearn.model_selection.GridSearchCV(MeanRF, RFparameters, verbose = 2)
# GSMeanRF.fit(Meandata, labels)
# MeansRFcores = GSMeanRF.best_score_
# print("Mean imputer, RF",MeansRFcores)
# GSMeanRFpred = GSMeanRF.best_estimator_
# GSMeanRFres = GSMeanRFpred.predict(Meandata)
# print(sklearn.metrics.confusion_matrix(labels, GSMeanRFres))
# print(sklearn.metrics.classification_report(labels, GSMeanRFres))

#with sep Mean imputer

SepMeanRF = sklearn.ensemble.RandomForestClassifier()
GSSepMeanRF = sklearn.model_selection.GridSearchCV(SepMeanRF, RFparameters, verbose = 2)
GSSepMeanRF.fit(SepMeandata, SepMeanlabels)
SepMeanRFcores = GSSepMeanRF.best_score_
print("Sep Mean imputer, RF",SepMeanRFcores)
print("Best params",GSSepMeanRF.best_params_)
GSSepMeanRFpred = GSSepMeanRF.best_estimator_
print(SepMeandata.columns)
print(GSSepMeanRFpred.feature_importances_)
GSSepMeanRFres = GSSepMeanRFpred.predict(SepMeandata)
print(sklearn.metrics.confusion_matrix(SepMeanlabels, GSSepMeanRFres))
print(sklearn.metrics.classification_report(SepMeanlabels, GSSepMeanRFres))

#with sep KNN imputer

# SepKNNRF = sklearn.ensemble.RandomForestClassifier()
# GSSepKNNRF = sklearn.model_selection.GridSearchCV(SepKNNRF, RFparameters, verbose = 2)
# GSSepKNNRF.fit(SepKNNdata, SepKNNlabels)
# SepKNNRFcores = GSSepKNNRF.best_score_
# print("Sep KNN imputer, RF",SepKNNRFcores)
# GSSepKNNRFpred = GSSepKNNRF.best_estimator_
# GSSepKNNRFres = GSSepKNNRFpred.predict(SepKNNdata)
# print(sklearn.metrics.confusion_matrix(SepKNNlabels, GSSepKNNRFres))
# print(sklearn.metrics.classification_report(SepKNNlabels, GSSepKNNRFres))

#----------------------------------------------------------------------------------------------------------------------#

#SVC

SVCparameters = {'C' : [0.1, 1, 10], 'kernel' : ['linear', 'poly', 'sigmoid']}

#with KNN imputer

# KNNSVC = sklearn.svm.SVC()
# GSKNNSVC = sklearn.model_selection.GridSearchCV(KNNSVC, SVCparameters, verbose = 2)
# GSKNNSVC.fit(KNNdata, labels)
# KNNSVCscores = GSKNNSVC.best_score_
# print("KNN imputer, SVC", KNNSVCscores)
# GSKNNSVCpred = GSKNNSVC.best_estimator_
# GSKNNSVCres = GSKNNSVCpred.predict(KNNdata)
# print(sklearn.metrics.confusion_matrix(labels, GSKNNSVCres))
# print(sklearn.metrics.classification_report(labels, GSKNNSVCres))

#with Mean imputer

# MeanSVC = sklearn.svm.SVC()
# GSMeanSVC = sklearn.model_selection.GridSearchCV(MeanSVC, SVCparameters, verbose = 2)
# GSMeanSVC.fit(Meandata, labels)
# MeanSVCscores = GSMeanSVC.best_score_
# print("Mean imputer, SVC", MeanSVCscores)
# GSMeanSVCpred = GSMeanSVC.best_estimator_
# GSMeanSVCres = GSMeanSVCpred.predict(Meandata)
# print(sklearn.metrics.confusion_matrix(labels, GSMeanSVCres))
# print(sklearn.metrics.classification_report(labels, GSMeanSVCres))

#with sep Mean imputer

# SepMeanSVC = sklearn.svm.SVC()
# GSSepMeanSVC = sklearn.model_selection.GridSearchCV(SepMeanSVC, SVCparameters, verbose = 2)
# GSSepMeanSVC.fit(SepMeandata, SepMeanlabels)
# SepMeanSVCscores = GSSepMeanSVC.best_score_
# print("Sep Mean imputer, SVC", SepMeanSVCscores)
# GSSepMeanSVCpred = GSSepMeanSVC.best_estimator_
# GSSepMeanSVCres = GSSepMeanSVCpred.predict(SepMeandata)
# print(sklearn.metrics.confusion_matrix(SepMeanlabels, GSSepMeanSVCres))
# print(sklearn.metrics.classification_report(SepMeanlabels, GSSepMeanSVCres))

#with sep KNN imputer

# SepKNNSVC = sklearn.svm.SVC()
# GSSepKNNSVC = sklearn.model_selection.GridSearchCV(SepKNNSVC, SVCparameters, verbose = 2)
# GSSepKNNSVC.fit(SepKNNdata, SepKNNlabels)
# SepKNNSVCscores = GSSepKNNSVC.best_score_
# print("Sep KNN imputer, SVC", SepKNNSVCscores)
# GSSepKNNSVCpred = GSSepKNNSVC.best_estimator_
# GSSepKNNSVCres = GSSepKNNSVCpred.predict(SepKNNdata)
# print(sklearn.metrics.confusion_matrix(SepKNNlabels, GSSepKNNSVCres))
# print(sklearn.metrics.classification_report(SepKNNlabels, GSSepKNNSVCres))

#----------------------------------------------------------------------------------------------------------------------#

#Adaboost

ABparameters = {'n_estimators' : [125, 150, 175], 'learning_rate' : [0.4, 0.5, 0.6]}

#with KNN imputer

# KNNAB = sklearn.ensemble.AdaBoostClassifier()
# GSKNNAB = sklearn.model_selection.GridSearchCV(KNNAB, ABparameters, verbose = 2)
# GSKNNAB.fit(KNNdata, labels)
# KNNABscores = GSKNNAB.best_score_
# print("KNN imputer, AB", KNNABscores)
# GSKNNABpred = GSKNNAB.best_estimator_
# GSKNNABres = GSKNNABpred.predict(KNNdata)
# print(sklearn.metrics.confusion_matrix(labels, GSKNNABres))
# print(sklearn.metrics.classification_report(labels, GSKNNABres))

#with Mean imputer

# MeanAB = sklearn.ensemble.AdaBoostClassifier()
# GSMeanAB = sklearn.model_selection.GridSearchCV(MeanAB, ABparameters, verbose = 2)
# GSMeanAB.fit(Meandata, labels)
# MeanABscores = GSMeanAB.best_score_
# print("Mean imputer, AB", MeanABscores)
# GSMeanABpred = GSMeanAB.best_estimator_
# GSMeanABres = GSMeanABpred.predict(Meandata)
# print(sklearn.metrics.confusion_matrix(labels, GSMeanABres))
# print(sklearn.metrics.classification_report(labels, GSMeanABres))

#with sep Mean imputer

# SepMeanAB = sklearn.ensemble.AdaBoostClassifier()
# GSSepMeanAB = sklearn.model_selection.GridSearchCV(SepMeanAB, ABparameters, verbose = 2)
# GSSepMeanAB.fit(SepMeandata, SepMeanlabels)
# SepMeanABscores = GSSepMeanAB.best_score_
# print("Sep Mean imputer, AB", SepMeanABscores)
# print("Best params", GSSepMeanAB.best_params_)
# GSSepMeanABpred = GSSepMeanAB.best_estimator_
# GSSepMeanABres = GSSepMeanABpred.predict(SepMeandata)
# print(sklearn.metrics.confusion_matrix(SepMeanlabels, GSSepMeanABres))
# print(sklearn.metrics.classification_report(SepMeanlabels, GSSepMeanABres))

#with sep KNN imputer

# SepKNNAB = sklearn.ensemble.AdaBoostClassifier()
# GSSepKNNAB = sklearn.model_selection.GridSearchCV(SepKNNAB, ABparameters, verbose = 2)
# GSSepKNNAB.fit(SepKNNdata, SepKNNlabels)
# SepKNNABscores = GSSepKNNAB.best_score_
# print("Sep KNN imputer, AB", SepKNNABscores)
# GSSepKNNABpred = GSSepKNNAB.best_estimator_
# GSSepKNNABres = GSSepKNNABpred.predict(SepKNNdata)
# print(sklearn.metrics.confusion_matrix(SepKNNlabels, GSSepKNNABres))
# print(sklearn.metrics.classification_report(SepKNNlabels, GSSepKNNABres))

#----------------------------------------------------------------------------------------------------------------------#

#Gaussian Naive Bayes

#with KNN imputer

# KNNGNB = sklearn.naive_bayes.GaussianNB()
# KNNGNBscores = sklearn. model_selection.cross_val_score(KNNGNB, KNNdata, labels , cv=5)
# # print("KNN imputer, GNB",KNNGNBscores.mean())
# KNNGNBres = KNNGNB.predict(KNNdata)
# print(sklearn.metrics.confusion_matrix(labels,KNNGNBres))
# print(sklearn.metrics.classification_report(labels, KNNGNBres))

#with Mean imputer

# MeanGNB = sklearn.naive_bayes.GaussianNB()
# MeanGNBscores = sklearn. model_selection.cross_val_score(MeanGNB, Meandata, labels , cv=5)
# print("Mean imputer, GNB",MeanGNBscores.mean())
# MeanGNBres = MeanGNB.predict(Meandata)
# print(sklearn.metrics.confusion_matrix(labels,MeanGNBres))
# print(sklearn.metrics.classification_report(labels, MeanGNBres))

#with sep Mean imputer

# SepMeanGNB = sklearn.naive_bayes.GaussianNB()
# SepMeanGNBscores = sklearn. model_selection.cross_val_score(SepMeanGNB, SepMeandata, SepMeanlabels , cv=5)
# print("SepMean imputer, GNB",SepMeanGNBscores.mean())
# SepMeanGNBres = SepMeanGNB.predict(SepMeandata)
# print(sklearn.metrics.confusion_matrix(SepMeanlabels,SepMeanGNBres))
# print(sklearn.metrics.classification_report(SepMeanlabels, SepMeanGNBres))

#with sep KNN imputer

# SepKNNGNB = sklearn.naive_bayes.GaussianNB()
# SepKNNGNBscores = sklearn. model_selection.cross_val_score(SepKNNGNB, SepKNNdata, SepKNNlabels , cv=5)
# print("Sep KNN imputer, GNB",SepKNNGNBscores.mean())
# SepKNNGNB.fit(SepKNNdata, SepKNNlabels)
# SepKNNGNBres = SepKNNGNB.predict(SepKNNdata)
# print(sklearn.metrics.confusion_matrix(SepKNNlabels,SepKNNGNBres))
# print(sklearn.metrics.classification_report(SepKNNlabels, SepKNNGNBres))

#----------------------------------------------------------------------------------------------------------------------#

#Stacking with best examples

#only with sep Mean inputer

# estimators = [
#     ('RF', sklearn.ensemble.RandomForestClassifier(n_estimators=750, max_depth=11, min_samples_leaf=1)),
#     ('AB', sklearn.ensemble.AdaBoostClassifier(learning_rate=0.5, n_estimators=125))
# ]
#
# stack = sklearn.ensemble.StackingClassifier(estimators=estimators, final_estimator=sklearn.linear_model.LogisticRegression())
# SepMeanStackscores = sklearn. model_selection.cross_val_score(stack, SepMeandata, SepMeanlabels , cv=5, verbose = 1)
# print("SepMean imputer, RF",SepMeanStackscores.mean())

#----------------------------------------------------------------------------------------------------------------------#

#DeepLearning

# NN = sklearn.neural_network.MLPClassifier(
#     hidden_layer_sizes = (9, 7, 6, 4, 2),
#     activation = 'logistic',
#     solver = 'adam',
#     learning_rate = 'adaptive',
#     verbose = 1
# )
# SepMeanNNscores = sklearn. model_selection.cross_val_score(NN, SepMeandata, SepMeanlabels , cv=5)
# print("SepMean imputer, GNB",SepMeanNNscores.mean())
