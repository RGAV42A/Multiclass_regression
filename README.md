# Multiclass_regression

Car Evaluation Data Set (https://www.kaggle.com/elikplim/car-evaluation-data-set) was analyzed. All factors and the target are categorical. 
There are 3 steps of the analysis 1) extract, transform, load (ETL), feature selection, and model selection in ETL_feature_selection_model_selection.py.  The data are loaded and cleaned. Then The factors are transformed with OneHotEncoder, and the target with LabelEncoder. The feature selection was performed with SelectKBest. The performance of score functions  f_classif and mutual_info_classif were compared. The performance of 6 models was compared LogisticRegression, LinearDiscriminantAnalysis, KNeighborsClassifier, DecisionTreeClassifier, GaussianNB, and SVМ. The best performance showed Support Vector Machine for classification with f_classif score function.

The second step was the evaluation of 4 ensemble methods - AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, and ExtraTreesClassifier in enseble_methods.py. 
The evaluation of the ensemble methods was combined with feature selection and hyperparameter tuning. At once the best combination of an ensemble model, the number of relevant features, and hyperparameters were performed. Very good performance and stability showed the GradientBoostingClassifier.

The third step was the model finalization in multiclass_reg.py. The performance and stability of the model were verified. The results from the analysis are shown in the graph of the confusion matrix.
