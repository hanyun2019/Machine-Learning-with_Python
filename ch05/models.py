# Introduction to Machine Learning with Python
# Chapter 5: Model evaluation and improvement
# Refreshed by Haowen Huang

from preamble import *

import mglearn
import numpy as np
import pandas as pd

from sklearn.datasets import make_blobs
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics.scorer import SCORERS

print("\n----------- Model evaluation and improvement -----------")

# create a synthetic dataset
X, y = make_blobs(random_state=0)
# split data and labels into a training and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# instantiate a model and fit it to the training set
logreg = LogisticRegression(solver='lbfgs',multi_class='auto').fit(X_train, y_train)
# evaluate the model on the test set
print("\nTest set score: {:.2f}".format(logreg.score(X_test, y_test)))
# Result: Test set score: 0.88      # when choose:  logreg = LogisticRegression().fit(X_train, y_train)     
# Result: Test set score: 0.92      # when choose:  logreg = LogisticRegression(solver='lbfgs',max_iter=5000).fit(X_train, y_train)
# Result: Test set score: 0.88      # when choose:  logreg = LogisticRegression(solver='lbfgs',multi_class='auto').fit(X_train, y_train)


## 5.1 Cross-Validation
# mglearn.plots.plot_cross_validation()

## 5.1.1 Cross-Validation in scikit-learn

print("\n----------- Model evaluation and improvement: Cross-Validation in scikit-learn -----------")

# from sklearn.model_selection import cross_val_score
# from sklearn.datasets import load_iris
# from sklearn.linear_model import LogisticRegression

iris = load_iris()
logreg = LogisticRegression()
# logreg = LogisticRegression(solver='lbfgs',multi_class='auto')
# Cross-validation is implemented in scikit-learn using the cross_val_score function from the model_selection module
scores = cross_val_score(logreg, iris.data, iris.target)
print("\nCross-validation scores: {}".format(scores))
# Result: Cross-validation scores: [0.961 0.922 0.958]      # when choose: logreg = LogisticRegression()
# Result: Cross-validation scores: [0.98  0.941 1.   ]      # when choose: logreg = LogisticRegression(solver='lbfgs',multi_class='auto')

# By default, cross_val_score performs three-fold cross-validation, returning three accuracy values.
# 缺省情况下，cross_val_score 执行3折交叉验证，返回3个精度值

# Change cv=5
scores = cross_val_score(logreg, iris.data, iris.target, cv=5)
print("\nCross-validation scores: {}".format(scores))
# Result: Cross-validation scores: [1.    0.967 0.933 0.9   1.   ]

print("\nAverage cross-validation score: {:.2f}".format(scores.mean()))
# Result: Average cross-validation score: 0.96

## 5.1.2 Benefits of Cross-Validation
# a) First, remember that train_test_split performs a random split of the data.
# b) Another benefit of cross-validation as compared to using a single split of the data is that we use our data more effectively.
# c) The main disadvantage of cross-validation is increased computational cost. 

## 5.1.3 Stratified K-Fold cross-validation and other strategies        分层k折交叉验证和其他策略

print("\n----------- Model evaluation and improvement: Stratified K-Fold cross-validation and other strategies -----------")

# from sklearn.datasets import load_iris

iris = load_iris()
print("\nIris labels:\n{}".format(iris.target))
# Iris labels:
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2]

# Imagine doing three-fold cross-validation on this dataset. 
# As the classes in training and test set would be different for all three splits, the three-fold cross-validation accuracy would be zero on this dataset. 
# As the simple k-fold strategy fails here, scikit-learn does not use k-fold for classification, but rather stratified k-fold cross-validation. 

# mglearn.plots.plot_stratified_cross_validation()



# More control over cross-validation
print("\n----------- Stratified K-Fold cross-validation and other strategies: More control over cross-validation -----------")

# from sklearn.model_selection import KFold

kfold = KFold(n_splits=5)
print("\nCross-validation scores:\n{}".format(cross_val_score(logreg, iris.data, iris.target, cv=kfold)))
# Result: 
# Cross-validation scores:
# [1.    0.933 0.433 0.967 0.433]

# This way, we can verify that it is indeed a really bad idea to use 3-fold (non-stratified) cross-validation on the iris dataset:
kfold = KFold(n_splits=3)
print("\nCross-validation scores - KFold(n_splits=3):\n{}".format(cross_val_score(logreg, iris.data, iris.target, cv=kfold)))
# Result:
# Cross-validation scores - KFold(n_splits=3):
# [0. 0. 0.]

# Remember: each fold corresponds to one of the classes, and so nothing can be learned. [specify again that this is on the iris dataset, it’s a little unclear]
# Another way to resolve this problem instead of stratifying the folds is to shuffle the data, to remove the ordering of the samples by label. 
# We can do that setting the shuffle parameter of KFold to True. 
# If we shuffle the data, we also need to fix the ran dom_state to get a reproducible shuffling. 
kfold = KFold(n_splits=3, shuffle=True, random_state=0)
print("\nCross-validation scores - KFold(n_folds=3, shuffle=True, random_state=0):\n{}".format(cross_val_score(logreg, iris.data, iris.target, cv=kfold)))
# Result:
# Cross-validation scores - KFold(n_folds=3, shuffle=True, random_state=0):
# [0.9  0.96 0.96]


# Leave-one-out cross-validation    留一法交叉验证
print("\n----------- Stratified K-Fold cross-validation and other strategies: Leave-one-out cross-validation -----------")

# from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
logreg = LogisticRegression(solver='lbfgs',multi_class='auto',max_iter=5000)    # Added by Haowen Huang to silent the warning

scores = cross_val_score(logreg, iris.data, iris.target, cv=loo)
print("Number of cv iterations: ", len(scores))
print("Mean accuracy: {:.2f}".format(scores.mean()))
# Result:
# Number of cv iterations:  150
# Mean accuracy: 0.97


# Shuffle-Split cross-validation    打乱划分交叉验证
print("\n----------- Stratified K-Fold cross-validation and other strategies: Shuffle-Split cross-validation -----------")

# mglearn.plots.plot_shuffle_split()

# from sklearn.model_selection import ShuffleSplit

# The following code splits the dataset into 50% training set and 50% test set for ten iterations:
shuffle_split = ShuffleSplit(test_size=.5, train_size=.5, n_splits=10)
scores = cross_val_score(logreg, iris.data, iris.target, cv=shuffle_split)
print("\nCross-validation scores:\n{}".format(scores))
# Result
# Cross-validation scores:
# [0.973 0.987 0.947 0.96  0.96  0.92  0.973 0.987 0.973 0.947]


# Cross-validation with groups  分组交叉验证
# To accurately evaluate the generalization to new faces, 
# we must therefore ensure that the training and test set contain images of different people.
print("\n----------- Stratified K-Fold cross-validation and other strategies: Cross-validation with groups -----------")

# mglearn.plots.plot_group_kfold()

# from sklearn.model_selection import GroupKFold

# create synthetic dataset
X, y = make_blobs(n_samples=12, random_state=0)
# assume the first three samples belong to the same group,
# then the next four, etc.
groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3]
scores = cross_val_score(logreg, X, y, groups, cv=GroupKFold(n_splits=3))
print("\nCross-validation scores:\n{}".format(scores))
# Result:
# Cross-validation scores:
# [0.75       0.6        0.66666667]




## 5.2 Grid Search
# Finding the values of the important parameters of a model (the ones that provide the best generalization performance) is a tricky task, 
# but necessary for almost all models and datasets.

# Consider the case of a kernel SVM with an RBF (radial basis function) kernel, as implemented in the SVC class. 
# There are two important parameters: the kernel bandwidth gamma and the regularization parameter C. 
# Say we want to try values 0.001, 0.01, 0.1, 1 and 10 for the parameter C, and the same for gamma. 
# Because we have six different settings for C and gamma that we want to try, we have 36 combinations of parameters in total.  
# The most commonly used method is grid search, which basically means trying all possible combinations of the parameters of interest.

## 5.2.1 Simple Grid Search
print("\n----------- Grid Search: Simple Grid Search -----------")

# naive grid search implementation

# from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)
print("Size of training set: {}   size of test set: {}".format(X_train.shape[0], X_test.shape[0]))

best_score = 0

for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        # for each combination of parameters, train an SVC
        svm = SVC(gamma=gamma, C=C)
        svm.fit(X_train, y_train)
        # evaluate the SVC on the test set
        score = svm.score(X_test, y_test)
        # if we got a better score, store the score and parameters
        if score > best_score:
            best_score = score
            best_parameters = {'C': C, 'gamma': gamma}

print("\nBest score: {:.2f}".format(best_score))
print("\nBest parameters: {}".format(best_parameters))
# Result:
# Size of training set: 112   size of test set: 38
# Best score: 0.97
# Best parameters: {'C': 100, 'gamma': 0.001}



## 5.2.2 The danger of overfitting the parameters and the validation set    参数过拟合的风险和验证集
# However, the above 97.3% accurate could be overly optimistic (or just wrong) for the following reason: 
# we tried many different parameters, and selected the one with best accuracy on the test set. 
# However, that doesn’t mean that this accuracy carries over to new data.
# We need an independent data set to evaluate, one that was not used to create the model.

# One way to resolve this problem is to split the data again, so we have three sets: 
# the training set to build the model, 
# the validation (or development) set to select the parameters of the model, 
# and the test set, to evaluate the performance of the selected parameters.

print("\n----------- Grid Search: The danger of overfitting the parameters and the validation set -----------")

# mglearn.plots.plot_threefold_split()

# from sklearn.svm import SVC
# split data into train+validation set and test set
X_trainval, X_test, y_trainval, y_test = train_test_split(
    iris.data, iris.target, random_state=0)
# split train+validation set into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(
    X_trainval, y_trainval, random_state=1)
print("\nSize of training set: {}   size of validation set: {}   size of test set:"
      " {}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))

best_score = 0

for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        # for each combination of parameters, train an SVC
        svm = SVC(gamma=gamma, C=C)
        svm.fit(X_train, y_train)
        # evaluate the SVC on the validation set
        score = svm.score(X_valid, y_valid)
        # if we got a better score, store the score and parameters
        if score > best_score:
            best_score = score
            best_parameters = {'C': C, 'gamma': gamma}

# rebuild a model on the combined training and validation set,
# and evaluate it on the test set
svm = SVC(**best_parameters)
svm.fit(X_trainval, y_trainval)
test_score = svm.score(X_test, y_test)
print("\nBest score on validation set: {:.2f}".format(best_score))
print("\nBest parameters: ", best_parameters)
print("\nTest set score with best parameters: {:.2f}".format(test_score))
# Result:
# Size of training set: 84   size of validation set: 28   size of test set: 38
# Best score on validation set: 0.96
# Best parameters:  {'C': 10, 'gamma': 0.001}
# Test set score with best parameters: 0.92



## 5.2.3 Grid Search with Cross-Validation  带交叉验证的网格搜索
# For a better estimate of the generalization performance, instead of using a single split into a training and a validation set, 
# we can use cross-validation to evaluate the performance of each parameter combination.
print("\n----------- Grid Search: Grid Search with Cross-Validation -----------")

for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        # for each combination of parameters,
        # train an SVC
        svm = SVC(gamma=gamma, C=C)
        # perform cross-validation
        scores = cross_val_score(svm, X_trainval, y_trainval, cv=5)
        # compute mean cross-validation accuracy
        score = np.mean(scores)
        # if we got a better score, store the score and parameters
        if score > best_score:
            best_score = score
            best_parameters = {'C': C, 'gamma': gamma}
# rebuild a model on the combined training and validation set
svm = SVC(**best_parameters)
svm.fit(X_trainval, y_trainval)
print("\nSVC(**best_parameters):\n",svm)
# Result:
# SVC(**best_parameters):
#  SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
#     max_iter=-1, probability=False, random_state=None, shrinking=True,
#     tol=0.001, verbose=False)

# mglearn.plots.plot_cross_val_selection()
# mglearn.plots.plot_grid_search_overview()

# Because grid-search with cross-validation is such a commonly used method to adjust parameters,
# scikit-learn provides the GridSearchCV class that implements it in the form of an estimator. 
# To use the GridSearchCV class, you first need to specify the parameters you want to search over using a dictionary. 
# GridSearchCV will then perform all the necessary model fits.
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
print("\nParameter grid:\n{}".format(param_grid))
# Result:
# Parameter grid:
# {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

# We can now instantiate the GridSearchCV class with the model SVC, the parameter grid to search param_grid, 
# and the cross-validation strategy we want to use, say 5 fold (stratified) cross-validation:

# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVC
grid_search = GridSearchCV(SVC(), param_grid, cv=5, return_train_score=True)
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)
grid_search.fit(X_train, y_train)
print("\nGridSearchCV configuration:\n",grid_search)
# Result: 
# GridSearchCV configuration:
#  GridSearchCV(cv=5, error_score='raise-deprecating',
#              estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#                            decision_function_shape='ovr', degree=3,
#                            gamma='auto_deprecated', kernel='rbf', max_iter=-1,
#                            probability=False, random_state=None, shrinking=True,
#                            tol=0.001, verbose=False),
#              iid='warn', n_jobs=None,
#              param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100],
#                          'gamma': [0.001, 0.01, 0.1, 1, 10, 100]},
#              pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
#              scoring=None, verbose=0)

print("\nTest set score: {:.2f}".format(grid_search.score(X_test, y_test)))
# Result: Test set score: 0.97

print("\nBest parameters: {}".format(grid_search.best_params_))
# Result: Best parameters: {'C': 100, 'gamma': 0.01}
print("\nBest cross-validation score: {:.2f}".format(grid_search.best_score_))
# Result: Best cross-validation score: 0.97


print("\nBest estimator:\n{}".format(grid_search.best_estimator_))
# Result:
# Best estimator:
# SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
#     max_iter=-1, probability=False, random_state=None, shrinking=True,
#     tol=0.001, verbose=False)


# Analyzing the result of cross-validation
print("\n----------- Grid Search: Grid Search with Cross-Validation - Analyzing the result of cross-validation -----------")

# import pandas as pd
# convert to Dataframe
results = pd.DataFrame(grid_search.cv_results_)
# show the first 5 rows
# display(results.head())
print("\nresults:\n",results.head())
# results:
#     mean_fit_time  std_fit_time  mean_score_time  std_score_time  ...  \
# 0       5.56e-04      3.04e-05         2.61e-04        1.79e-05  ...   
# 1       5.19e-04      1.09e-05         2.42e-04        1.77e-06  ...   
# 2       5.59e-04      7.29e-05         2.58e-04        2.44e-05  ...   
# 3       5.26e-04      1.63e-05         2.42e-04        2.64e-06  ...   
# 4       5.92e-04      1.10e-04         3.28e-04        1.62e-04  ...   

#   split3_train_score split4_train_score mean_train_score  std_train_score  
# 0               0.37               0.36             0.37         2.85e-03  
# 1               0.37               0.36             0.37         2.85e-03  
# 2               0.37               0.36             0.37         2.85e-03  
# 3               0.37               0.36             0.37         2.85e-03  
# 4               0.37               0.36             0.37         2.85e-03  

# [5 rows x 22 columns]

scores = np.array(results.mean_test_score).reshape(6, 6)
print("\nnp.array(results.mean_test_score).reshape(6, 6):\n",scores)
# Result:
# np.array(results.mean_test_score).reshape(6, 6):
#  [[0.366 0.366 0.366 0.366 0.366 0.366]
#  [0.366 0.366 0.366 0.366 0.366 0.366]
#  [0.366 0.696 0.92  0.955 0.366 0.366]
#  [0.696 0.929 0.964 0.946 0.92  0.509]
#  [0.929 0.964 0.964 0.938 0.92  0.571]
#  [0.964 0.973 0.955 0.946 0.92  0.571]]

# plot the mean cross-validation scores
# mglearn.tools.heatmap(scores, xlabel='gamma', xticklabels=param_grid['gamma'],
#                       ylabel='C', yticklabels=param_grid['C'], cmap="viridis")



# 在非网格的空间中搜索
print("\n----------- Grid Search: Grid Search with Cross-Validation - 在非网格的空间中搜索 -----------")

param_grid = [{'kernel': ['rbf'],
               'C': [0.001, 0.01, 0.1, 1, 10, 100],
               'gamma': [0.001, 0.01, 0.1, 1, 10, 100]},
              {'kernel': ['linear'],
               'C': [0.001, 0.01, 0.1, 1, 10, 100]}]
print("\nList of grids:\n{}".format(param_grid))
# List of grids:
# [{'kernel': ['rbf'], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}, {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100]}]

grid_search = GridSearchCV(SVC(), param_grid, cv=5, return_train_score=True)
grid_search.fit(X_train, y_train)
print("\nBest parameters: \n{}".format(grid_search.best_params_))
# Result:
# Best parameters: 
# {'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}

print("\nBest cross-validation score: \n{:.2f}".format(grid_search.best_score_))
# Result:
# Best cross-validation score: 
# 0.97

print("\nGridSearchCV configuration:\n",grid_search)
# Result:
# GridSearchCV configuration:
#  GridSearchCV(cv=5, error_score='raise-deprecating',
#              estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#                            decision_function_shape='ovr', degree=3,
#                            gamma='auto_deprecated', kernel='rbf', max_iter=-1,
#                            probability=False, random_state=None, shrinking=True,
#                            tol=0.001, verbose=False),
#              iid='warn', n_jobs=None,
#              param_grid=[{'C': [0.001, 0.01, 0.1, 1, 10, 100],
#                           'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
#                           'kernel': ['rbf']},
#                          {'C': [0.001, 0.01, 0.1, 1, 10, 100],
#                           'kernel': ['linear']}],
#              pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
#              scoring=None, verbose=0)

results = pd.DataFrame(grid_search.cv_results_)
# we display the transposed table so that it better fits on the page:
# display(results.T)


# Using different cross-validation strategies with grid search
print("\n----------- Grid Search: Grid Search with Cross-Validation - Using different cross-validation strategies with grid search -----------")

# Similarly to cross_val_score, GridSearchCV uses stratified k-fold cross-validation by default for classification, and k-fold cross-validation for regression. 
# However, you can also pass any cross-validation splitter, as described in section XX as the cv parameter in GridSearchCV.
# In particular, to get only a single split into a training and validation set, you can use ShuffleSplit or StratifiedShuffleSplit with n_iter=1. 
# This might be helpful for very large datasets, or very slow models.

# Nested cross-validation   嵌套交叉验证
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
scores = cross_val_score(GridSearchCV(SVC(), param_grid, cv=5),
                         iris.data, iris.target, cv=5)
print("\nCross-validation scores:\n", scores)
# Result:
# Cross-validation scores:
#  [0.967 1.    0.967 0.967 1.   ]

print("\nMean cross-validation score:\n", scores.mean())
# Result:
# Mean cross-validation score:
#  0.9800000000000001

def nested_cv(X, y, inner_cv, outer_cv, Classifier, parameter_grid):
    outer_scores = []
    # for each split of the data in the outer cross-validation
    # (split method returns indices of training and test parts)
    for training_samples, test_samples in outer_cv.split(X, y):
        # find best parameter using inner cross-validation
        best_parms = {}
        best_score = -np.inf
        # iterate over parameters
        for parameters in parameter_grid:
            # accumulate score over inner splits
            cv_scores = []
            # iterate over inner cross-validation
            for inner_train, inner_test in inner_cv.split(
                    X[training_samples], y[training_samples]):
                # build classifier given parameters and training data
                clf = Classifier(**parameters)
                clf.fit(X[inner_train], y[inner_train])
                # evaluate on inner test set
                score = clf.score(X[inner_test], y[inner_test])
                cv_scores.append(score)
            # compute mean score over inner folds
            mean_score = np.mean(cv_scores)
            if mean_score > best_score:
                # if better than so far, remember parameters
                best_score = mean_score
                best_params = parameters
        # build classifier on best parameters using outer training set
        clf = Classifier(**best_params)
        clf.fit(X[training_samples], y[training_samples])
        # evaluate
        outer_scores.append(clf.score(X[test_samples], y[test_samples]))
    return np.array(outer_scores)

# from sklearn.model_selection import ParameterGrid, StratifiedKFold
scores = nested_cv(iris.data, iris.target, StratifiedKFold(5), StratifiedKFold(5), SVC, ParameterGrid(param_grid))
print("\nCross-validation scores: \n{}".format(scores))
# Result:
# Cross-validation scores: 
# [0.967 1.    0.967 0.967 1.   ]




## 5.3 Evaluation Metrics and Scoring
print("\n----------- Evaluation Metrics and Scoring -----------")

## 5.3.1 Keep the End Goal in Mind

## 5.3.2 Metrics for Binary Classification      二分类指标
# Kinds of errors
# Imbalanced datasets       不平衡数据集
print("\n----------- Evaluation Metrics and Scoring: Metrics for Binary Classification -----------")

# from sklearn.datasets import load_digits
digits = load_digits()
y = digits.target == 9
print("\ndigits.data.shape:\n",digits.data.shape)
print("\ndigits.data:\n",digits.data)
print("\ny.shape:\n",y.shape)
print("\ny:\n",y)
print("\ndigits.target.shape:\n",digits.target.shape)
print("\ndigits.target:\n",digits.target)
# Result:
# digits.data.shape:
#  (1797, 64)
#
# digits.data:
#  [[ 0.  0.  5. ...  0.  0.  0.]
#  [ 0.  0.  0. ... 10.  0.  0.]
#  [ 0.  0.  0. ... 16.  9.  0.]
#  ...
#  [ 0.  0.  1. ...  6.  0.  0.]
#  [ 0.  0.  2. ... 12.  0.  0.]
#  [ 0.  0. 10. ... 12.  1.  0.]]
#
# y.shape:
#  (1797,)
#
# y:
#  [False False False ... False  True False]
#
# digits.target.shape:
#  (1797,)
#
# digits.target:
#  [0 1 2 ... 8 9 8]


X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state=0)

# from sklearn.dummy import DummyClassifier
dummy_majority = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
pred_most_frequent = dummy_majority.predict(X_test)
print("\nDummyClassifier configuration:\n",dummy_majority)
print("\nUnique predicted labels: {}".format(np.unique(pred_most_frequent)))
print("\nTest score(DummyClassifier): {:.2f}".format(dummy_majority.score(X_test, y_test)))
# Result:
# DummyClassifier configuration:
#  DummyClassifier(constant=None, random_state=None, strategy='most_frequent')
#
# Unique predicted labels: [False]
# Test score(DummyClassifier): 0.90


# from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
print("\nDecisionTreeClassifier configuration:\n",tree)
pred_tree = tree.predict(X_test)
print("\nTest score(DecisionTreeClassifier): {:.2f}".format(tree.score(X_test, y_test)))
# Result:
# DecisionTreeClassifier configuration:
#  DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=2,
#                        max_features=None, max_leaf_nodes=None,
#                        min_impurity_decrease=0.0, min_impurity_split=None,
#                        min_samples_leaf=1, min_samples_split=2,
#                        min_weight_fraction_leaf=0.0, presort=False,
#                        random_state=None, splitter='best')
#
# Test score(DecisionTreeClassifier): 0.92


# from sklearn.linear_model import LogisticRegression
dummy = DummyClassifier().fit(X_train, y_train)
pred_dummy = dummy.predict(X_test)
print("\ndummy classifier score: {:.2f}".format(dummy.score(X_test, y_test)))
# dummy classifier score: 0.82

logreg = LogisticRegression(C=0.1).fit(X_train, y_train)
pred_logreg = logreg.predict(X_test)
print("\nlogistic regression score: {:.2f}".format(logreg.score(X_test, y_test)))
# logistic regression score: 0.98

# Clearly accuracy is an inadequate measure to quantify predictive performance in this imbalanced setting. 
# For the rest of this chapter, we will explore alternative metrics that provide better guidance in selecting models. 
# In particular, we would like to have metrics that tell us how much better a model is than making “most frequent” predictions or random predictions, 
# as they are computed in pred_most_frequent and pred_dummy. 
# If we use a metric to assess our models, it should definitely be able to weed out these nonsense predictions.


# Confusion matrices    混淆矩阵
print("\n----------- Evaluation Metrics and Scoring: Confusion matrices -----------")

# from sklearn.metrics import confusion_matrix

# Let’s inspect the predictions of LogisticRegression above using the confusion_matrix function. 
# We already stored the predictions on the test set in pred_logreg.
confusion = confusion_matrix(y_test, pred_logreg)
print("\nConfusion matrix:\n{}".format(confusion))
# Confusion matrix:
# [[401   2]
#  [  8  39]]

# The output of confusion_matrix is a two by two array, 
# where the rows correspond to the true classes, 
# and the columns corresponds to the predicted classes. 
# Each entry counts for how many data points in the class given by the row the prediction was the class given by the column.

# mglearn.plots.plot_confusion_matrix_illustration()
# [[401   2]    the rows correspond to the true classes: rows0 - true 'not nine' / row1 - true 'nine' 
#  [  8  39]]   the columns corresponds to the predicted classes: column0 - predicted 'not nine' / predicted 'nine'

# [[TN   FP]
#  [FN   TP]]

print("\nMost frequent class:")
print(confusion_matrix(y_test, pred_most_frequent))
print("\nDummy model:")
print(confusion_matrix(y_test, pred_dummy))
print("\nDecision tree:")
print(confusion_matrix(y_test, pred_tree))
print("\nLogistic Regression")
print(confusion_matrix(y_test, pred_logreg))
# Most frequent class:
# [[403   0]
#  [ 47   0]]

# Dummy model:
# [[366  37]
#  [ 44   3]]

# Decision tree:
# [[390  13]
#  [ 24  23]]

# Logistic Regression
# [[401   2]
#  [  8  39]]

# 精度: Accuracy = (TP + TN) / (TP + TN + FP + FN)
# Accuracy is the number of correct prediction (TP and TN) divided by the number of all samples.

# 准确率: Precision = TP / (TP + FP)
# Precision measures how many of the samples predicted as positive are actually positive.

# 召回率: Recall = TP / (TP + FN) 
# Recall measures how many of the positive samples are captured by the positive predictions.

# There is a trade-off between optimizing recall and optimizing precision. 
# You can trivially(微不足道地) obtain a perfect recall if you predict all samples to belong to the positive class - there will be no false negatives, and no true negatives either. 
# However, predicting all samples as positive will result in many false positives, therefore the precision will be very low. 
# On the other hand, if you find a model that predicts only the single data point it is most sure about as positive, and the rest as negative, 
# then precision will be perfect, but recall will be very bad.

# f-分数: F = 2 * (precision * recall)/(precision + recall)
# This particular variant is also known as the f_1-score. 
# As it takes precision and recall into account, it can be a better measure than accuracy on imbalanced binary classification datasets.

# from sklearn.metrics import f1_score
print("\nf1 score most frequent: {:.2f}".format(
    f1_score(y_test, pred_most_frequent)))
print("\nf1 score dummy: {:.2f}".format(f1_score(y_test, pred_dummy)))
print("\nf1 score tree: {:.2f}".format(f1_score(y_test, pred_tree)))
print("\nf1 score logistic regression: {:.2f}".format(
    f1_score(y_test, pred_logreg)))
# f1 score most frequent: 0.00
# f1 score dummy: 0.10
# f1 score tree: 0.55
# f1 score logistic regression: 0.89

# from sklearn.metrics import classification_report
print("\npred_most_frequent report:\n")
print(classification_report(y_test, pred_most_frequent, target_names=["not nine", "nine"]))
print("\npred_dummy report:\n")
print(classification_report(y_test, pred_dummy, target_names=["not nine", "nine"]))
print("\npred_logreg report:\n")
print(classification_report(y_test, pred_logreg, target_names=["not nine", "nine"]))
# pred_most_frequent report:

#               precision    recall  f1-score   support

#     not nine       0.90      1.00      0.94       403
#         nine       0.00      0.00      0.00        47

#     accuracy                           0.90       450
#    macro avg       0.45      0.50      0.47       450
# weighted avg       0.80      0.90      0.85       450


# pred_dummy report:

#               precision    recall  f1-score   support

#     not nine       0.90      0.91      0.90       403
#         nine       0.14      0.13      0.13        47

#     accuracy                           0.82       450
#    macro avg       0.52      0.52      0.52       450
# weighted avg       0.82      0.82      0.82       450


# pred_logreg report:

#               precision    recall  f1-score   support

#     not nine       0.98      1.00      0.99       403
#         nine       0.95      0.83      0.89        47

#     accuracy                           0.98       450
#    macro avg       0.97      0.91      0.94       450
# weighted avg       0.98      0.98      0.98       450


# Taking uncertainty into account
# The confusion matrix and the classification report provide a very detailed analysis of a particular set of predictions. 
# However, the predictions themselves already threw away a lot of information that is contained in the model. 
# As we discusses in Chapter 2, most classifiers provide a decision_function or a predict_proba method to assess degrees of certainty about predictions.

# Making predictions can be seen as thresholding the output of decision_function or predict_proba at a certain fixed point - 
# in binary classification zero for the decision function and 0.5 for predict_proba.

# Below is an example of an imbalanced binary classification task, with 400 blue points classified against 50 red points.

print("\n----------- Evaluation Metrics and Scoring: Taking uncertainty into account -----------")

X, y = make_blobs(n_samples=(400, 50), cluster_std=[7.0, 2],
                  random_state=22)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
svc = SVC(gamma=.05).fit(X_train, y_train)

# mglearn.plots.plot_decision_threshold()
print("\nsvc_predict report(threshold > 0):\n")
print(classification_report(y_test, svc.predict(X_test)))
y_pred_lower_threshold = svc.decision_function(X_test) > -.8
print("\nsvc_predict report(threshold > -0.8):\n")
print(classification_report(y_test, y_pred_lower_threshold))
# svc_predict report(threshold > 0):

#               precision    recall  f1-score   support

#            0       0.97      0.89      0.93       104
#            1       0.35      0.67      0.46         9

#     accuracy                           0.88       113
#    macro avg       0.66      0.78      0.70       113
# weighted avg       0.92      0.88      0.89       113


# svc_predict report(threshold > -0.8):

#               precision    recall  f1-score   support

#            0       1.00      0.82      0.90       104
#            1       0.32      1.00      0.49         9

#     accuracy                           0.83       113
#    macro avg       0.66      0.91      0.69       113
# weighted avg       0.95      0.83      0.87       113



# Precision-Recall curves   准确率-召回率曲线
# As we just discussed, changing the threshold that is used to make a classification decision in a model is 
# a way to adjust the trade-off of precision and recall for a given classifier. 

print("\n----------- Evaluation Metrics and Scoring: Precision-Recall curves -----------")

print("\n----------- Precision-Recall curves: SVC(gamma=0.05) -----------")

# from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(
    y_test, svc.decision_function(X_test))

# Use more data points for a smoother curve
X, y = make_blobs(n_samples=(4000, 500), cluster_std=[7.0, 2], random_state=22)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
svc = SVC(gamma=.05).fit(X_train, y_train)
precision, recall, thresholds = precision_recall_curve(
    y_test, svc.decision_function(X_test))
# find threshold closest to zero
close_zero = np.argmin(np.abs(thresholds))
# plt.plot(precision[close_zero], recall[close_zero], 'o', markersize=10,
#          label="threshold zero", fillstyle="none", c='k', mew=2)

# plt.plot(precision, recall, label="precision recall curve")
# plt.xlabel("Precision")
# plt.ylabel("Recall")
# plt.legend(loc="best")


# Different classifiers can work well in different parts of the curve, that is at different operating points. 
# Below we compare the SVC we trained to a random forest trained on the same dataset.
print("\n----------- Precision-Recall curves: Compare SVC & Random forest -----------")

# from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=0, max_features=2)
rf.fit(X_train, y_train)

# RandomForestClassifier has predict_proba, but not decision_function
precision_rf, recall_rf, thresholds_rf = precision_recall_curve(
    y_test, rf.predict_proba(X_test)[:, 1])

# plt.plot(precision, recall, label="svc")

# plt.plot(precision[close_zero], recall[close_zero], 'o', markersize=10,
#          label="threshold zero svc", fillstyle="none", c='k', mew=2)

# plt.plot(precision_rf, recall_rf, label="rf")

close_default_rf = np.argmin(np.abs(thresholds_rf - 0.5))
# plt.plot(precision_rf[close_default_rf], recall_rf[close_default_rf], '^', c='k',
#          markersize=10, label="threshold 0.5 rf", fillstyle="none", mew=2)
# plt.xlabel("Precision")
# plt.ylabel("Recall")
# plt.legend(loc="best")

# From the comparison plot we can see that the random forest performs better at the extremes, for very high recall or very high precision requirements. 
# Around the middle (around precision=0.7), the SVM performance better. 
# If we only looked at the f1-score to compare overall performance, we would have missed these subtleties. 
# The f1-score only captures one point on the precision-recall curve, the one given by the default threshold:

print("f1_score of random forest: {:.3f}".format(
    f1_score(y_test, rf.predict(X_test))))
print("f1_score of svc: {:.3f}".format(f1_score(y_test, svc.predict(X_test))))

from sklearn.metrics import average_precision_score
ap_rf = average_precision_score(y_test, rf.predict_proba(X_test)[:, 1])
ap_svc = average_precision_score(y_test, svc.decision_function(X_test))
print("Average precision of random forest: {:.3f}".format(ap_rf))
print("Average precision of svc: {:.3f}".format(ap_svc))
# f1_score of random forest: 0.610
# f1_score of svc: 0.656
# Average precision of random forest: 0.660
# Average precision of svc: 0.666

# Comparing two precision-recall curves provides a lot of detailed insight, but is a fairly manual process. 
# For automatic model comparison, we might want to summarize the information contained in the curve, without limiting ourselves to a particular threshold or operating point.
# One particular way to summarize the precision-recall curve by computing the integral or area under the curve of the precision-recall curve, 
# also known as average precision.

# from sklearn.metrics import average_precision_score

ap_rf = average_precision_score(y_test, rf.predict_proba(X_test)[:, 1])
ap_svc = average_precision_score(y_test, svc.decision_function(X_test))
print("\nAverage precision of random forest: {:.3f}".format(ap_rf))
print("\nAverage precision of svc: {:.3f}".format(ap_svc))
# Average precision of random forest: 0.660
# Average precision of svc: 0.666

# When averaging over all possible thresholds, we see that random forest and SVC perform similarly well. 
# This is quite different than the result we got from f1_score above.


# Receiver Operating Characteristics (ROC) and Area Under the Curve(AUC)  受试者工作特征(ROC)和AUC
print("\n----------- Receiver Operating Characteristics (ROC) and AUC -----------")

# Instead of reporting precision and recall, ROC shows the false positive rate FPR(false positive rate) against the true positive rate TPR. 
# FPR = FP/(FP + TN)
# TPR(Recall) = TP/(TP + FN)

# from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, svc.decision_function(X_test))

# plt.plot(fpr, tpr, label="ROC Curve")
# plt.xlabel("FPR")
# plt.ylabel("TPR (recall)")
# find threshold closest to zero
close_zero = np.argmin(np.abs(thresholds))
# plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=10,
#          label="threshold zero", fillstyle="none", c='k', mew=2)
# plt.legend(loc=4)

# For the ROC curve, the ideal curve is close to the top left: you want a classifier that produces a high recall while keeping a low false positive rate. 
# Compared to the default threshold of zero, the curve shows that we could achieve a significant higher recall (around 0.9) while only increasing the FPR slightly. 
# The point closes to the top left might be a better operating point than the one chosen by default. 
# Again, be aware that choosing a threshold should not be done on the test set, but on a separate validation set.

# You can find a comparison of the Random Forest and the SVC using ROC curves in Figure roc_curve_comparison.
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])

# plt.plot(fpr, tpr, label="ROC Curve SVC")
# plt.plot(fpr_rf, tpr_rf, label="ROC Curve RF")

# plt.xlabel("FPR")
# plt.ylabel("TPR (recall)")
# plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=10,
#          label="threshold zero SVC", fillstyle="none", c='k', mew=2)
close_default_rf = np.argmin(np.abs(thresholds_rf - 0.5))
# plt.plot(fpr_rf[close_default_rf], tpr[close_default_rf], '^', markersize=10,
#          label="threshold 0.5 RF", fillstyle="none", c='k', mew=2)

# plt.legend(loc=4)


# As for the precision-recall curve, we often want to summarize the ROC curve using a single number, the area under the curve. 
# Often the area under the ROC-curve is just called AUC (area under the curve) and it is understood that the curve in question is the ROC curve. 
# We can compute the area under the ROC curve using the roc_auc_score function:

# from sklearn.metrics import roc_auc_score
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
svc_auc = roc_auc_score(y_test, svc.decision_function(X_test))
print("\nAUC for Random Forest: {:.3f}".format(rf_auc))
print("\nAUC for SVC: {:.3f}".format(svc_auc))
# AUC for Random Forest: 0.937
# AUC for SVC: 0.916

# For classification problems with imbalanced classes, using AUC for model-selection is often much more meaningful than using accuracy. 
# Let’s go back to the problem we studied above of classifying all nines in the digits dataset versus all other digits. 
# We will classify the dataset with an SVM with three different settings of the kernel band‐width gamma:

y = digits.target == 9

X_train, X_test, y_train, y_test = train_test_split(
    digits.data, y, random_state=0)

# plt.figure()

for gamma in [1, 0.05, 0.01]:
    svc = SVC(gamma=gamma).fit(X_train, y_train)
    accuracy = svc.score(X_test, y_test)
    auc = roc_auc_score(y_test, svc.decision_function(X_test))
    fpr, tpr, _ = roc_curve(y_test , svc.decision_function(X_test))
    print("\ngamma = {:.2f}  accuracy = {:.2f}  AUC = {:.2f}".format(
          gamma, accuracy, auc))
    # plt.plot(fpr, tpr, label="gamma={:.3f}".format(gamma))
# plt.xlabel("FPR")
# plt.ylabel("TPR")
# plt.xlim(-0.01, 1)
# plt.ylim(0, 1.02)
# plt.legend(loc="best")

# Result:
# gamma = 1.00  accuracy = 0.90  AUC = 0.50
# gamma = 0.05  accuracy = 0.90  AUC = 1.00
# gamma = 0.01  accuracy = 0.90  AUC = 1.00

# The accuracy of all three settings of gamma is the same, 90%. 
# Finally with gamma=0.01, we get a perfect AUC of 1.0. 
# That means that all positive points are ranked higher than all negative points according to the decision function. 
# In other words, with the right threshold, this model can classify the data perfectly! 

# For this reason, we highly recommend using AUC when evaluating models on imbalanced data. 
# Keep in mind that AUC does not make use of the default threshold, 
# ]so adjusting the decision threshold might be necessary to obtain useful classification results from a model with high AUC.



## 5.3.3 Multi-class classification
# Imagine a three-class classification problem with 85% of points belonging to class A, 10% belonging to class B and 5% belonging to class C. 
# What does being 85% accurate mean on this dataset?
print("\n----------- Multi-class classification -----------")

# Apart from accuracy, common tools are the confusion matrix and the classification report we saw in the binary case above.
# Let’s apply these two detailed evaluation methods on the task of classifying the 10 different hand-written digits in the digits dataset.

# from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state=0)
lr = LogisticRegression().fit(X_train, y_train)
pred = lr.predict(X_test)
print("\ndigits.data.shape", digits.data.shape)
print("\ndigits.target.shape", digits.target.shape)
# digits.data.shape (1797, 64)
# digits.target.shape (1797,)

print("\ndigits.data", digits.data)
print("\ndigits.target", digits.target)
# digits.data [[ 0.  0.  5. ...  0.  0.  0.]
#  [ 0.  0.  0. ... 10.  0.  0.]
#  [ 0.  0.  0. ... 16.  9.  0.]
#  ...
#  [ 0.  0.  1. ...  6.  0.  0.]
#  [ 0.  0.  2. ... 12.  0.  0.]
#  [ 0.  0. 10. ... 12.  1.  0.]]

# digits.target [0 1 2 ... 8 9 8]

print("\ndigits.data[:2]", digits.data[:2])
print("\ndigits.target[:20]", digits.target[:20])
# digits.data[:2] [[ 0.  0.  5. 13.  9.  1.  0.  0.  0.  0. 13. 15. 10. 15.  5.  0.  0.  3.
#   15.  2.  0. 11.  8.  0.  0.  4. 12.  0.  0.  8.  8.  0.  0.  5.  8.  0.
#    0.  9.  8.  0.  0.  4. 11.  0.  1. 12.  7.  0.  0.  2. 14.  5. 10. 12.
#    0.  0.  0.  0.  6. 13. 10.  0.  0.  0.]
#  [ 0.  0.  0. 12. 13.  5.  0.  0.  0.  0.  0. 11. 16.  9.  0.  0.  0.  0.
#    3. 15. 16.  6.  0.  0.  0.  7. 15. 16. 16.  2.  0.  0.  0.  0.  1. 16.
#   16.  3.  0.  0.  0.  0.  1. 16. 16.  6.  0.  0.  0.  0.  1. 16. 16.  6.
#    0.  0.  0.  0.  0. 11. 16. 10.  0.  0.]]

# digits.target[:20] [0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9]


print("\nAccuracy: {:.3f}".format(accuracy_score(y_test, pred)))
print("\nConfusion matrix:\n{}".format(confusion_matrix(y_test, pred)))
# Accuracy: 0.953
# Confusion matrix:
# [[37  0  0  0  0  0  0  0  0  0]
#  [ 0 39  0  0  0  0  2  0  2  0]
#  [ 0  0 41  3  0  0  0  0  0  0]
#  [ 0  0  1 43  0  0  0  0  0  1]
#  [ 0  0  0  0 38  0  0  0  0  0]
#  [ 0  1  0  0  0 47  0  0  0  0]
#  [ 0  0  0  0  0  0 52  0  0  0]
#  [ 0  1  0  1  1  0  0 45  0  0]
#  [ 0  3  1  0  0  0  0  0 43  1]
#  [ 0  0  0  1  0  1  0  0  1 44]]

# The model has an accuracy of 95.3%, which already tells us that we are doing pretty well. 
# The confusion matrix provides us with some more detail. 
# As for the binary case, each row corresponds to a true label, and each column corresponds to a predicted label.

# scores_image = mglearn.tools.heatmap(
#     confusion_matrix(y_test, pred), xlabel='Predicted label',
#     ylabel='True label', xticklabels=digits.target_names,
#     yticklabels=digits.target_names, cmap=plt.cm.gray_r, fmt="%d")
# plt.title("Confusion matrix")
# plt.gca().invert_yaxis()

# For the first class, the digit 0, there are 37 samples in the class, 
# 
# and all of these samples were classified as class 0 (no false negatives for the zero class).
# We can see that because all other entries in the first row of the confusion matrix are zero. 
# 
# We can also see that no other digits was mistakenly classified as zero, 
# because all other entries in the first column of the confusion matrix are zero (no false positives for class zero).

# With the classification_report function, we can compute the precision, recall and f-score for each class:
print(classification_report(y_test, pred))
#                 precision   recall   f1-score   support

#            0       1.00      1.00      1.00        37
#            1       0.89      0.91      0.90        43
#            2       0.95      0.93      0.94        44
#            3       0.90      0.96      0.92        45
#            4       0.97      1.00      0.99        38
#            5       0.98      0.98      0.98        48
#            6       0.96      1.00      0.98        52
#            7       1.00      0.94      0.97        48
#            8       0.93      0.90      0.91        48
#            9       0.96      0.94      0.95        47

#     accuracy                           0.95       450
#    macro avg       0.95      0.95      0.95       450
# weighted avg       0.95      0.95      0.95       450

# “macro” averaging computes the unweighted the per-class f-scores. This gives equal weight to all classes, no matter what their size is.
# “weighted” averaging computes the mean of the per-class f-scores, weighted by their support. This is what is reported in the classification report.
# “micro” averaging computes total number of false positives, false negatives and true positives over all classes, and then compute precision, recall and f-score using these counts.

print("Micro average f1 score: {:.3f}".format(
    f1_score(y_test, pred, average="micro")))
print("Macro average f1 score: {:.3f}".format(
    f1_score(y_test, pred, average="macro")))
# Micro average f1 score: 0.953
# Macro average f1 score: 0.954



## 5.3.4 Regression metrics
# Evaluation for regression can be done in similar detail as we did for classification above, 
# for example by analyzing over-predicting the target versus under-predicting the target. 
# However, in most application we’ve seen, using the default R^2 used in the score method of all regressors is enough. 



## 5.3.5  Using evaluation metrics in model selection
print("\n----------- Using evaluation metrics in model selection -----------")

# we often want to use metrics like AUC in model selection using Grid SearchCV or cross_val_score.
# Luckily scikit-learn provides a very simple way to achieve this, via the scoring argument that can be used in both GridSearchCV and cross_val_score. 
# For example, we want to evaluate the SVC classifier on the “nine vs rest” task on the digits dataset, using the AUC score. C
# hanging the score from the default (accuracy) to AUC can be done by providing "roc_auc" as the scoring parameter:
# default scoring for classification is accuracy

print("\n----------- Using evaluation metrics in model selection: SVC Classifier -----------")

print("\nDefault scoring: {}".format(cross_val_score(SVC(), digits.data, digits.target == 9, cv=5)))
# providing scoring="accuracy" doesn't change the results
explicit_accuracy =  cross_val_score(SVC(), digits.data, digits.target == 9, scoring="accuracy", cv=5)
print("\nExplicit accuracy scoring: {}".format(explicit_accuracy))
roc_auc =  cross_val_score(SVC(), digits.data, digits.target == 9, scoring="roc_auc", cv=5)
print("\nAUC scoring: {}".format(roc_auc))
# Default scoring: [0.9 0.9 0.9 0.9 0.9]
# Explicit accuracy scoring: [0.9 0.9 0.9 0.9 0.9]
# AUC scoring: [0.997 0.997 0.996 0.998 0.992]


print("\n----------- Using evaluation metrics in model selection: GridSearchCV-----------")
# Similarly we can change the metric used to pick the best parameters in GridSearchCV:
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target == 9, random_state=0)

# we provide a somewhat bad grid to illustrate the point:
param_grid = {'gamma': [0.0001, 0.01, 0.1, 1, 10]}
# using the default scoring of accuracy:
grid = GridSearchCV(SVC(), param_grid=param_grid)
grid.fit(X_train, y_train)
print("\nGrid-Search with default scoring accuracy:")
print("\nBest parameters:", grid.best_params_)
print("\nBest cross-validation score (accuracy)): {:.3f}".format(grid.best_score_))
print("\nTest set AUC: {:.3f}".format(
    roc_auc_score(y_test, grid.decision_function(X_test))))
print("\nTest set accuracy: {:.3f}".format(grid.score(X_test, y_test)))
# Grid-Search with default scoring of accuracy:
# Best parameters: {'gamma': 0.0001}
# Best cross-validation score (accuracy)): 0.970
# Test set AUC: 0.992
# Test set accuracy: 0.973

# using AUC scoring instead:
grid = GridSearchCV(SVC(), param_grid=param_grid, scoring="roc_auc")
grid.fit(X_train, y_train)
print("\nGrid-Search with AUC:")
print("\nBest parameters:", grid.best_params_)
print("\nBest cross-validation score (AUC): {:.3f}".format(grid.best_score_))
print("\nTest set AUC: {:.3f}".format(
    roc_auc_score(y_test, grid.decision_function(X_test))))
print("\nTest set accuracy: {:.3f}".format(grid.score(X_test, y_test)))
# Grid-Search with AUC:
# Best parameters: {'gamma': 0.01}
# Best cross-validation score (AUC): 0.997
# Test set AUC: 1.000
# Test set accuracy: 1.000

# When using accuracy, the parameter gamma=0.0001 is selected, while gamma=0.01 is selected when using AUC. 
# The cross-validation accuracy is consistent with the test set accuracy in both cases. 
# However, using AUC found a better parameter setting, both in terms of AUC and even in terms of accuracy.

# The most important values for the scoring parameter for classification are accuracy (the default), 
# roc_auc for the area under the ROC curve, average_precision for the area under the precision-recall curve, 
# f1, f1_macro, f1_micro and f1_weighted for the binary F1 score and the different weighted variants.

# For regression, the most commonly used values are r2 for the R^2 score, mean_squared_error for mean squared error 
# and mean_absolute_error for mean absolute error.

# You can find a full list of supported arguments in the documentation
# http://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules
# or by looking at the SCORER dictionary defined in the metrics.scorer module:

# from sklearn.metrics.scorer import SCORERS
print("\nAvailable scorers:")
print(sorted(SCORERS.keys()))
# Available scorers:
# ['accuracy', 'adjusted_mutual_info_score', 'adjusted_rand_score', 'average_precision', 'balanced_accuracy', 'brier_score_loss', 'completeness_score', 'explained_variance', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'fowlkes_mallows_score', 'homogeneity_score', 'jaccard', 'jaccard_macro', 'jaccard_micro', 'jaccard_samples', 'jaccard_weighted', 'max_error', 'mutual_info_score', 'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error', 'normalized_mutual_info_score', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc', 'v_measure_score']



## 5.4 Summary and outlook
# In this chapter we discussed cross-validation, grid-search and evaluation metrics, the corner-stones of evaluating and improving machine learning algorithms. 

# There are two particular points that we made in this chapter:

# 1) Cross-validation
# We therefore need to resort to a split into training data for model building, validation data for model and parameter selection, and test data for model evaluation. 
# Instead of a simple split, we can replace each of these splits with cross-validation. 
# The most commonly used form as described above is a train-test split for evaluation, 
# and using cross-validation on the training set for model and parameter selection.

# 2) evaluation metric or scoring function used for model selection and model evaluation
# The techniques model evaluation and selection techniques we described so far are the most important tools in a data scientists toolbox. 

# However, grid search and cross validation as we described it in this chapter can only be applied to a single supervised model. 
# We have seen before, however, that many models require preprocessing, and that in some applications, 
# like the face recognition example in Chapter 3, extracting a different representation of the data can be useful. 

# In the next chapter, we will introduce the Pipeline class, 
# which allows us to use grid-search and cross-validation on these complex chains of algorithms.