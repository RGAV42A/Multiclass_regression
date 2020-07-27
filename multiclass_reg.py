import pandas as pd
import numpy as np
#from pylab import *
'''
import matplotlib.pyplot as plt
#from tempfile import TemporaryFile
import random,lxml,xlrd,sys,json
#from lxml.html import parse
import urllib,re,requests
#from lxml import objectify
#from pandas.io.json import json_normalize
#from pandas.io.json import loads
#from pandas.io.pytables import HDFStore
import pickle,random
#from sqlalchemy import create_engine
from numpy import nan as NA
#from pandas_datareader import data as web # da instaliram na moja komp
# import pandas.io.data as web
#import fix_yahoo_finance # lipsva
import datetime,sqlite3
#import pandas.io.sql as sql
#from sklearn.neural_network import *
#from sklearn.preprocessing import LabelEncoder
from sklearn import datasets
#from sklearn import preprocessing
##from sklearn import linear_model,feature_selection
#from sklearn.preprocessing import PolynomialFeatures
#from sklearn.pipeline import make_pipeline
#import statsmodels.graphics.api as smg
from statsmodels.stats.outliers_influence import variance_inflation_factor,OLSInfluence
import statsmodels.api as sm
import statsmodels.stats.diagnostic as ssd
##from scipy.optimize import curve_fit
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNC
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import cdist, pdist
'''


### Multiclass Logistic Regression  ###
## load data
iris = datasets.load_iris()
X = iris.data
y = iris.target

'''
##  Normalize Data
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)
# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)
#### Training Logistic Regression Model
# l1 regularization gives better results
lr = LogisticRegression(penalty='l1', C=10, random_state=0)
lr.fit(X_train, y_train)
# generate evaluation metrics
print("Train - Accuracy :", metrics.accuracy_score(y_train, lr.predict(X_train)))
print("Train - Confusion matrix :",metrics.confusion_matrix(y_train,lr.predict(X_train)))
print("Train - classification report :", metrics.classification_report(y_train, lr.predict(X_train)))
print("Test - Accuracy :", metrics.accuracy_score(y_test, lr.predict(X_test)))
print("Test - Confusion matrix :",metrics.confusion_matrix(y_test,lr.predict(X_test)))
print("Test - classification report :", metrics.classification_report(y_test, lr.predict(X_test)))
'''








