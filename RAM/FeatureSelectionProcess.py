__author__ = 'hafiz'
# Feature Importance
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
# load the iris datasets
filename = 'rawdata/features/featurerankingset.csv'
mydata = pd.read_csv('rawdata/features/featurerankingset.csv')
target = mydata["Label"]
data = np.genfromtxt(filename, delimiter=',')[1:,:-1]
#select all but the last column as data
# data = mydata.ix[:,:-1]
# print data
# dataset = datasets.load_iris()
labels = np.genfromtxt(filename, delimiter=',', usecols=-1, dtype=str)
# print dataset
# print dataset.data
# fit an Extra Trees model to the data
# model = ExtraTreesClassifier()
# model.fit(data, target)
# # display the relative importance of each attribute
# print(model.feature_importances_)
model = RandomForestClassifier()
for i in range(1,20):
    model.fit(data, target)
    print(model.feature_importances_)
    # print (model.oob_score_)
    # print (model.oob_decision_function_)

