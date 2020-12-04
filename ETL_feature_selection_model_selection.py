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
from sklearn.model_selection import cross_val_score


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
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=200)

results = []
logloss_resu = []
names = []

# Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto',probability=True)))



# define the pipeline to evaluate
for name,model in models:
    fs = SelectKBest()
    pipeline = Pipeline(steps=[('anova',fs), ('model', model)])
    # define the grid
    grid = {'anova__k':[i+1 for i in range(X.shape[1])],'anova__score_func':[f_classif,mutual_info_classif]}
    # define the grid search
    search = RandomizedSearchCV(pipeline, grid, scoring='neg_log_loss', n_jobs=-1, cv=cv)
    # perform the search
    result = search.fit(X, y)

    print('\nModel name:',name)

    means = search.cv_results_['mean_fit_time']
    stds = search.cv_results_['std_fit_time']
    params = search.cv_results_['mean_score_time']
    timem = search.cv_results_['std_score_time']

    # measure calculation time
    #for mean, stdev, param in zip(means, stds, params):
        #print("mean_fit_time:%f std_fit_time:%f  mean_score_time:%f std_score_time:%f" % (mean.sum(), stdev.sum(), param.sum(),timem.sum()))

    logloss_results = cross_val_score(search.best_estimator_,X,y, cv=cv,scoring='neg_log_loss')
    # convert scores to positive
    logloss_results = np.absolute(logloss_results)
    logloss_resu.append(logloss_results)

    cv_results = cross_val_score(search.best_estimator_,X,y, cv=cv,scoring='accuracy')
    results.append(cv_results)

    names.append(name)
    # summarize best

    print('Best Mean Log-loss: %.3f' % result.best_score_)
    print('Best Config: %s' % result.best_params_)
    msg = "accuracy -> %s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(121)
pyplot.boxplot(results)
ax.set_xticklabels(names)
ax = fig.add_subplot(122)
pyplot.boxplot(logloss_resu)
ax.set_xticklabels(names)
pyplot.savefig('feature_model_selection.png',dpi=80)

'''
Model name: LR
Best Mean Log-loss: -0.278
Best Config: {'anova__score_func': <function f_classif at 0x7f6cfb8522f0>, 'anova__k': 17}
accuracy -> LR: 0.875576 (0.008148)
Model name: LDA

Best Mean Log-loss: -0.300
Best Config: {'anova__score_func': <function f_classif at 0x7f6cfb8522f0>, 'anova__k': 18}
accuracy -> LDA: 0.894289 (0.011175)
Model name: KNN

Best Mean Log-loss: -0.343
Best Config: {'anova__score_func': <function f_classif at 0x7f6cfb8522f0>, 'anova__k': 18}
accuracy -> KNN: 0.911274 (0.021974)
Model name: CART

Best Mean Log-loss: -0.415
Best Config: {'anova__score_func': <function mutual_info_classif at 0x7f6cf950a158>, 'anova__k': 9}
accuracy -> CART: 0.848578 (0.029430)
Model name: NB

Best Mean Log-loss: -4.689
Best Config: {'anova__score_func': <function f_classif at 0x7f6cfb8522f0>, 'anova__k': 5}
accuracy -> NB: 0.696957 (0.022538)
Model name: SVM

Best Mean Log-loss: -0.123  -> very slow
Best Config: {'anova__score_func': <function f_classif at 0x7f6cfb8522f0>, 'anova__k': 17}
accuracy -> SVM: 0.936919 (0.012119)
'''
