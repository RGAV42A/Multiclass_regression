import subprocess

out = subprocess.run(['/bin/bash', '-c','dir'],shell=True)
print(out)


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

# ensembles
ensembles = []
ensembles.append(('AB', AdaBoostClassifier()))
ensembles.append(('GBM', GradientBoostingClassifier()))
ensembles.append(('RF', RandomForestClassifier(n_estimators=10)))
ensembles.append(('ET', ExtraTreesClassifier(n_estimators=10)))
results = []
names = []
for name, model in ensembles:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    cv_results = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Ensemble Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.savefig('ensemble_methods.png',dpi=80)


