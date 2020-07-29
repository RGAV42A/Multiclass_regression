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
from sklearn.feature_selection import RFE

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
        df[new_name]=df[rfactors[ix]]*df[rfactors[iy]]


#Y=df['buying']
#X=df[rfactors]

Y=df.iloc[:,0]
X=df.iloc[:,1:22]

## FIT MODEL
#lr=sm.MNLogit(Y,X).fit()
#print(lr.summary())
#print(lr.pvalues)
'''
### SPILIT DATAFRAME
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7,random_state=35)

#### Multiclass Regression Model  with SelectFromModel  ????  threshold_  ????
lr = LogisticRegression(penalty='l2', C=1, random_state=30,solver='newton-cg',multi_class='multinomial')
selector = feature_selection.SelectFromModel(estimator=lr).fit(x_train, y_train)
resu = selector.get_support()
print(selector.threshold_)
step2factors=[]
for ix in range(len(X.columns)):
    #print(X.columns[ix],resu[ix])
    if resu[ix] == True:
        step2factors.append(X.columns[ix])
print(step2factors)

Y=df['buying']
X=df[step2factors]
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7,random_state=35)
mlr = LogisticRegression(penalty='l2', C=1, random_state=30,solver='newton-cg',multi_class='multinomial')
mlrm = mlr.fit(x_train, y_train)
# generate evaluation metrics
print("Train - Accuracy :", metrics.accuracy_score(y_train, mlrm.predict(x_train)))
#print("Train - Confusion matrix :",metrics.confusion_matrix(y_train,lr.predict(x_train)))
#print("Train - classification report :", metrics.classification_report(y_train, lr.predict(x_train)))
print("Test - Accuracy :", metrics.accuracy_score(y_test, mlrm.predict(x_test)))
#print("Test - Confusion matrix :",metrics.confusion_matrix(y_test,lr.predict(x_test)))
#print("Test - classification report :", metrics.classification_report(y_test, lr.predict(x_test)))

###  Naive Bayes classifier for categorical features

#clf = naive_bayes.MultinomialNB()
#clf.fit(X, Y)
#resu = clf.score(X,Y)
#print(resu)

### ## FEATURE SELECTION WITH RFE
listnames=X.columns.values
print(listnames)
r_sq=0
#Variable to store the optimum features
nof=0
#no of features
nof_list=np.arange(1,len(X.columns)+1)

###
for n in range(len(nof_list)):
    #X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3, random_state = 0)
    model = LogisticRegression(penalty='l2', C=1, random_state=30,solver='newton-cg',multi_class='multinomial')
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X,Y)
    X_test_rfe = rfe.transform(X)
    model.fit(X_train_rfe,Y)
    score = metrics.accuracy_score(Y, model.predict(X_test_rfe))
    #adjRsq=1-((1-score)*(X_test_rfe.shape[0]-1)/(X_test_rfe.shape[0]-n-1))
    print('{} - nof:{} - score:{:.4f}'.format(listnames[n],n,score))
    if(score>r_sq):
        r_sq = score
        nof = nof_list[n]
        temp = pd.Series(rfe.support_,index = listnames)
        selected_features_rfe = temp[temp==True].index
        print(selected_features_rfe)

print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, r_sq))
print(selected_features_rfe)
'''
#############
## plot resuals
rlist =['maint', 'doors', 'persons', 'lug_boot', 'safety','acceptability', 'maint_persons', 'maint_safety','maint_acceptability', 'persons_lug_boot', 'persons_safety','persons_acceptability', 'lug_boot_safety', 'safety_acceptability']
Y=df['buying']
X=df[rlist]
mlr = LogisticRegression(penalty='l2', C=1, random_state=30,solver='newton-cg',multi_class='multinomial')
mlrm = mlr.fit(X,Y)
y_pred=mlr.predict(X)
print(mlr.score(X,Y))
resid=Y-y_pred

### clear Outliers
res_ser = pd.Series(resid,index = Y.index.values)
filt_res= res_ser[abs(res_ser)>1.5]
filt_res=filt_res.index.values
df.drop([ix for ix in filt_res], inplace = True)
###

rlist =['maint', 'doors', 'persons', 'lug_boot', 'safety','acceptability', 'maint_persons', 'maint_safety','maint_acceptability', 'persons_lug_boot', 'persons_safety','persons_acceptability', 'lug_boot_safety', 'safety_acceptability']
Y=df['buying']
X=df[rlist]
mlr = LogisticRegression(penalty='l2', C=1, random_state=30,solver='newton-cg',multi_class='multinomial')
mlrm = mlr.fit(X,Y)
y_pred=mlr.predict(X)
print(mlr.score(X,Y))
resid=Y-y_pred


####  Histogram of standardized deviance residuals
plt.close('all')
fig, ax = plt.subplots()
ax.hist(resid, bins=7)
ax.set_title('Histogram of residuals')
fig.savefig('Histogram.png',dpi=125)

print('~~~~~~~')
