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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from matplotlib import pyplot


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


# define the evaluation method
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

# Spot-Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# define the pipeline to evaluate
for name,model in models:
    fs = SelectKBest(score_func=f_classif)
    pipeline = Pipeline(steps=[('anova',fs), ('model', model)])
    # define the grid
    grid = dict()
    grid['anova__k'] = [i+1 for i in range(X.shape[1])]
    # define the grid search
    search = RandomizedSearchCV(pipeline, grid, scoring='accuracy', n_jobs=-1, cv=cv)
    # perform the search
    result = search.fit(X, y)
    # summarize best
    print('Model name:',name)
    print('Best Mean Accuracy: %.3f' % result.best_score_)
    print('Best Config: %s' % result.best_params_)

# Model name: LR  -> Best Mean Accuracy: 0.878  -> Best Config: {'anova__k': 21}
# Model name: LDA -> Best Mean Accuracy: 0.892 -> Best Config: {'anova__k': 19}
# Model name: KNN -> Best Mean Accuracy: 0.928 -> Best Config: {'anova__k': 17}
# Model name: CART -> Best Mean Accuracy: 0.965 -> Best Config: {'anova__k': 17}
# Model name: NB -> Best Mean Accuracy: 0.803 -> Best Config: {'anova__k': 19}
# Model name: SVM -> Best Mean Accuracy: 0.935 -> Best Config: {'anova__k': 16}
