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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from scipy.stats import randint
from sklearn.pipeline import Pipeline
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

# ensembles
ensembles = []
ensembles.append(('AB', AdaBoostClassifier()))
ensembles.append(('GBM', GradientBoostingClassifier()))
ensembles.append(('RF', RandomForestClassifier()))
ensembles.append(('ET', ExtraTreesClassifier()))
results = []
logloss_resu = []
names = []
for name, model in ensembles:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=200)
    fs = SelectKBest(score_func=f_classif)
    pipeline = Pipeline(steps=[('anova',fs), ('model', model)])
    # define the grid
    grid = {'anova__k':[i+1 for i in range(X.shape[1])],'model__n_estimators':randint(10,400)}
    # define the grid search
    search = RandomizedSearchCV(pipeline, grid, scoring='neg_log_loss', n_jobs=-1, cv=cv, random_state=500)
    search.fit(X, y.ravel())
    print('\n',name)
    print ('Best Parameters: ', search.best_params_)

    cv_results = cross_val_score(search.best_estimator_, X, y, cv=cv, scoring='accuracy')
    logloss_results = cross_val_score(search.best_estimator_, X, y, cv=cv, scoring='neg_log_loss')
    results.append(cv_results)
    logloss_resu.append(logloss_results)
    names.append(name)
    msg = "accuracy -> %s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    msg = "logloss -> %s: %f (%f)" % (name, logloss_results.mean(), logloss_results.std())
    print(msg)

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Ensemble Algorithm Comparison')
ax = fig.add_subplot(121)
ax.set_title('accuracy')
pyplot.boxplot(results)
ax.set_xticklabels(names)
ax = fig.add_subplot(122)
ax.set_title('neg_log_loss')
pyplot.boxplot(logloss_resu)
ax.set_xticklabels(names)
pyplot.tight_layout()
pyplot.savefig('ensemble_methods.png',dpi=80)
'''
AB
Best Parameters:  {'anova__k': 15, 'model__n_estimators': 297}
accuracy -> AB: 0.840480 (0.025328)
logloss -> AB: -0.520793 (0.128535)
 GBM
Best Parameters:  {'anova__k': 18, 'model__n_estimators': 223}
accuracy -> GBM: 0.982638 (0.004961)
logloss -> GBM: -0.050972 (0.009568)
 RF
Best Parameters:  {'anova__k': 18, 'model__n_estimators': 223}
accuracy -> RF: 0.955628 (0.011193)
logloss -> RF: -0.129185 (0.014370)
 ET
Best Parameters:  {'anova__k': 20, 'model__n_estimators': 151}
accuracy -> ET: 0.958718 (0.010394)
logloss -> ET: -0.131837 (0.011212)
'''