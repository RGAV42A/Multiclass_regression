# Multiclass_regression

Car Evaluation Data Set (https://www.kaggle.com/elikplim/car-evaluation-data-set) was analyzed. All factors and the target are categorical. 
There are 3 steps of the analysis 1) extract, transform, load (ETL), feature selection, and model selection in ETL_feature_selection_model_selection.py.  The data are loaded and cleaned. Then The factors are transformed with OneHotEncoder, and the target with LabelEncoder. The feature selection was performed with SelectKBest. The performance of score functions  f_classif and mutual_info_classif were compared. The performance of 6 models was compared LogisticRegression, LinearDiscriminantAnalysis, KNeighborsClassifier, DecisionTreeClassifier, GaussianNB, and SVC. 
