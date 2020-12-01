import subprocess

#out = subprocess.run(['/bin/bash', '-c','dir'],shell=True)
#print(out)


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from sklearn import metrics
import warnings

warnings.simplefilter('ignore')


##### READ DATA
col_names=['buying','maint','doors','persons','lug_boot','safety','acceptability']
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data',names = col_names)

print('df.head\n',df.head())

for ix in col_names:
    print(df[ix].unique())


# calculate duplicates
dups = df.duplicated()
# report if there are any duplicates
##print(dups.any())
# list all duplicate rows
##print(df[dups])

# delete duplicate rows
df.drop_duplicates(inplace=True)
print(df.shape)

# retrieve the array of data
data = df.values
# separate into input and output columns
X = data[:, :-1].astype(str)
y = data[:, -1].astype(str)
# define one hot encoding
encoder = OneHotEncoder(sparse=False)
# transform data
X = encoder.fit_transform(X)
y_encoder = LabelEncoder()
y = y_encoder.fit_transform(y)

# feature selection
def select_features(X, y):
    fs = SelectKBest(score_func=f_classif, k=17)
    fs.fit(X, y)
    X = fs.transform(X)
    return X
X = select_features(X,y)
print('select_features: Done')
# define the evaluation method
clf_DT_Bag = BaggingClassifier()
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
# specify parameters and distributions to sample from
param_dist = {'n_estimators':sp_randint(10,200),'bootstrap':['True','False'],'oob_score':['True','False']}
# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(clf_DT_Bag, param_distributions=param_dist,cv=cv, n_iter=n_iter_search,
verbose=0, n_jobs=-1, random_state=500,scoring='accuracy')
grid_result = random_search.fit(X, y)
print ('Best Parameters: ', random_search.best_params_)
results = cross_val_score(random_search.best_estimator_,X,y, cv=cv,scoring='accuracy')
print ("CART(Bagging) CV: ", results.mean(),results.std())

#Best Parameters:  {'bootstrap': 'True', 'n_estimators': 152, 'oob_score': 'True'}
#CART(Bagging) CV:  0.9731909189913712 0.009699470952365605

