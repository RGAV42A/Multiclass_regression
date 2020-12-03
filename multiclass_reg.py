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
from sklearn.ensemble import GradientBoostingClassifier
from matplotlib import pyplot
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

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

print('X.shape',X.shape)
# feature selection
def select_features(X, y):
    fs = SelectKBest(score_func=f_classif, k=18)
    fs.fit(X, y)
    X = fs.transform(X)
    return X

# define the evaluation method
X = select_features(X, y)
print('X.shape',X.shape)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=200)
model = GradientBoostingClassifier(n_estimators = 223)
model.fit(X,y)
cv_results = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
msg = "%s: %f (%f)" % ('GMB', cv_results.mean(), cv_results.std())
print(msg)

y_pred = model.predict(X)
multiclass = confusion_matrix(y, y_pred)
fig, ax = plot_confusion_matrix(conf_mat=multiclass,
                                colorbar=True,
                                show_absolute=False,
                                show_normed=True)
pyplot.savefig('multiclass_confusion_matrix.png',dpi=100)

# GMB: 0.964903 (0.012377)