# Introduction to Machine Learning with Python
# Chapter 5: Model evaluation and improvement
# Refreshed by Haowen Huang

from preamble import *

import mglearn
import numpy as np
import pandas as pd

from sklearn.datasets import make_blobs
from sklearn.datasets import load_iris

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






