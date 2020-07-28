import pandas as pd
import numpy as np
#from pylab import *
import matplotlib.pyplot as plt
from matplotlib import ticker
import urllib,re,requests
from numpy import nan as NA
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets
from sklearn import preprocessing
from sklearn import linear_model,feature_selection
#from sklearn.preprocessing import PolynomialFeatures
#from sklearn.pipeline import make_pipeline
import statsmodels.graphics.api as smg
from statsmodels.stats.outliers_influence import variance_inflation_factor,OLSInfluence
import statsmodels.api as sm
import statsmodels.stats.diagnostic as ssd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from statsmodels.formula.api import Logit
import statsmodels.api as sm
from matplotlib.colors import ListedColormap
from sklearn import naive_bayes


'''
### Multiclass Regression  ###
## load data
path='car_evaluation.csv'
names=['buying','maint','doors','persons','lug_boot','safety','acceptability']
dataframe=pd.read_csv(path,delimiter=',',names=names)

###  LABEL ENCODING

for ix in names:
    iy = dataframe[ix].unique()
    dataframe[ix].replace(iy, range(1,len(iy)+1), inplace = True)
dataframe.to_csv('car_eval_encoded.csv',index=False)
'''
#### LOAD ENCODED DATA

df=pd.read_csv('car_eval_encoded.csv')

### CALC INTERACTIONS
rfactors=['maint','doors','persons','lug_boot','safety','acceptability']
for ix in range(0,len(rfactors)-1):
    for iy in range(ix+1,len(rfactors)):
        new_name=str(rfactors[ix]+'_'+rfactors[iy])
        #print(new_name,type(new_name))
        df[new_name]=df[rfactors[ix]]*df[rfactors[iy]]
#Y=df['buying']
#X=df[rfactors]

Y=df.iloc[:,0]
X=df.iloc[:,1:22]

## FIT MODEL
lr=sm.MNLogit(Y,X).fit()
print(lr.summary())
#print(lr.pvalues)

### SPILIT DATAFRAME
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8,random_state=35)

#### Training Logistic Regression Model
lr = LogisticRegression(penalty='l2', C=1, random_state=30,solver='newton-cg',multi_class='multinomial')
lr.fit(x_train, y_train)
# generate evaluation metrics
print("Train - Accuracy :", metrics.accuracy_score(y_train, lr.predict(x_train)))
#print("Train - Confusion matrix :",metrics.confusion_matrix(y_train,lr.predict(x_train)))
#print("Train - classification report :", metrics.classification_report(y_train, lr.predict(x_train)))
print("Test - Accuracy :", metrics.accuracy_score(y_test, lr.predict(x_test)))
#print("Test - Confusion matrix :",metrics.confusion_matrix(y_test,lr.predict(x_test)))
#print("Test - classification report :", metrics.classification_report(y_test, lr.predict(x_test)))

###  Naive Bayes classifier for categorical features

clf = naive_bayes.MultinomialNB()
clf.fit(X, Y)
resu = clf.score(X,Y)
print(resu)






