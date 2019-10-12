# Introduction to Machine Learning with Python
# Chapter 6: Algorithm Chains and Pipelines
# Refreshed by Haowen Huang

from preamble import *

import mglearn
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures


# For example we noticed that we can greatly improve the performance of a kernel SVM on the cancer dataset by using the MinMaxScaler for preprocessing. 
# The code for splitting the data, computing minimum and maximum, scaling the data, and training the SVM is shown below:
# load and split the data
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

# compute minimum and maximum on the training data
scaler = MinMaxScaler().fit(X_train)

# rescale the training data
X_train_scaled = scaler.transform(X_train)

svm = SVC()
# learn an SVM on the scaled training data
svm.fit(X_train_scaled, y_train)
# scale the test data and score the scaled data
X_test_scaled = scaler.transform(X_test)
print("\nTest score: {:.2f}".format(svm.score(X_test_scaled, y_test)))
# Test score: 0.95


## 6.1 Parameter Selection with Preprocessing
print("\n----------- Parameter Selection with Preprocessing -----------")

# from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=5)

# for illustration purposes only, don't use this code!
# We then use the scaled training data to run our grid-search using cross-validation. 
# For each split in the cross-validation, some part of the original training set will be declared the training part of this split, and some the test part of the split. 
# The test part is used to measure how new data will look like to a model trained on the training part. 
# However, we already used the information contained in the test part of the split, when scaling the data. 
grid.fit(X_train_scaled, y_train)

print("\nBest cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("\nBest parameters: ", grid.best_params_)
print("\nTest set accuracy: {:.2f}".format(grid.score(X_test_scaled, y_test)))
# Best cross-validation accuracy: 0.98
# Best parameters:  {'C': 1, 'gamma': 1}
# Test set accuracy: 0.97

# So the splits in the cross-validation no longer correctly mirror how new data will look to the modeling process. 
# We already leaked information from these parts of the data into our modeling process. 
# This will lead to overly optimistic results during cross-validation, and possibly the selection of suboptimal parameters.

# To get around this problem, the splitting of the data set during cross-validation should be done before doing any preprocessing. 


## 6.2 Building Pipelines
# Let’s look at how we can use the Pipeline to express the work-flow for training an SVM after scaling the data MinMaxScaler, 
# for now without the grid-search.
print("\n----------- Building Pipelines -----------")

# from sklearn.pipeline import Pipeline

# Here, we created two steps, the first called "scaler" is a MinMaxScaler, 
# the second, called "svm" is an SVC. 
# Now, we can fit the pipeline, like any other scikit-learn estimator:
pipe = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC())])
pipe.fit(X_train, y_train)
print("\nPipeline configuration:\n",pipe)
# Here, pipe.fit first calls fit on the first step, the scaler, 
# then transforms the training data using the scaler, 
# and finally fits the SVM with the scaled data. 

# To evaluate on the test data, we simply call pipe.score:
print("\nTest score: {:.2f}".format(pipe.score(X_test, y_test)))
# Pipeline configuration:
#  Pipeline(memory=None,
#          steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))),
#                 ('svm',
#                  SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#                      decision_function_shape='ovr', degree=3,
#                      gamma='auto_deprecated', kernel='rbf', max_iter=-1,
#                      probability=False, random_state=None, shrinking=True,
#                      tol=0.001, verbose=False))],
#          verbose=False)

# Test score: 0.95


## 6.3 Using Pipelines in Grid-searches
# We define a parameter grid to search over, and construct a GridSearchCV from the pipeline and the parameter grid. 
# When specifying the parameter grid, there is a slight change, though. 
# We need to specify for each parameter which step of the pipeline it belongs to.
print("\n----------- Using Pipelines in Grid-searches -----------")

param_grid = {'svm__C': [0.001, 0.01, 0.1, 1, 10, 100],
              'svm__gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)

grid.fit(X_train, y_train)
print("\nBest cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("\nTest set score: {:.2f}".format(grid.score(X_test, y_test)))
print("\nBest parameters: {}".format(grid.best_params_))
# Best cross-validation accuracy: 0.98
# Test set score: 0.97
# Best parameters: {'svm__C': 1, 'svm__gamma': 1}

# In contrast to the grid-search we did before, now for each split in the cross-validation, the MinMaxScaler is refit with only the training splits, 
# not leaking any information of the test split into the parameter search.



## Illustrating information leakage
print("\n----------- Illustrating information leakage -----------")
# A great example of leaking information in cross-validation is given in Hastie et al. (《统计学习基础》) and we reproduce an adapted version here.
# Let us consider a synthetic(假想的) regression task with 100 samples and 1000 features that are sampled independently from a Gaussian distribution. 
# We also sample the response from a Gaussian distribution:
rnd = np.random.RandomState(seed=0)
X = rnd.normal(size=(100, 10000))
y = rnd.normal(size=(100,))
print("\nX.shape:",X.shape)
print("\nX:\n",X)
print("\ny.shape:\n",y.shape)
print("\ny:\n",y)
# X.shape: (100, 10000)
# X:
#  [[ 1.764  0.4    0.979 ...  0.517 -0.033  1.298]
#  [-0.202 -0.833  1.734 ... -0.057 -1.059 -0.327]
#  [ 0.33  -0.     0.818 ... -1.511  0.977  1.256]
#  ...
#  [ 1.421 -2.095  0.235 ...  0.457 -1.56  -0.248]
#  [-0.296  1.147 -1.498 ...  1.417  0.215 -0.856]
#  [-1.051  0.221  2.002 ... -0.797  0.875  1.372]]

# y.shape:
#  (100,)
# y:
#  [ 0.514  1.113  0.675 -0.561 -0.835  2.172  1.503  1.489  0.653  0.021
#  -0.4    0.957 -0.228  0.844  0.093 -0.007  0.556 -0.004  0.322  0.758
#   0.331  0.726 -0.521 -0.465 -1.307 -0.018 -0.599 -1.318 -0.578 -1.281
#   1.752  0.425 -2.091 -0.476 -0.487 -1.339  0.84   0.151  0.146 -0.116
#  -2.312 -0.667  0.778 -0.122 -1.302  0.636 -0.532  0.259 -1.191  0.334
#  -1.819 -0.472 -0.349  0.054  0.596  0.768  0.147  0.789  1.68   1.081
#   1.148 -0.605  0.604 -2.322  0.427 -1.86   0.166 -1.288 -0.932 -1.595
#   0.08   0.264  0.398 -1.139  1.623  0.584 -0.185  1.084  0.683  2.058
#   2.242 -2.081 -2.247  0.525 -0.031 -0.721 -0.519  0.272  0.304 -0.317
#  -0.795 -1.622  0.764  0.326 -0.813 -1.041 -0.442  0.626 -0.653  0.806]

# Given the way we created the dataset, there is no relation between the data X and the target y (they are independent), 
# so it should not be possible to learn anything from this data set.

# We will now do the following: 
# First select the most informative of the ten thousand features using SelectPercentile feature selection, 
# and then evaluate a Ridge regressor using cross-validation:

# from sklearn.feature_selection import SelectPercentile, f_regression
select = SelectPercentile(score_func=f_regression, percentile=5).fit(X, y)
# The information we leaked from the test-folds is here!!!
X_selected = select.transform(X)
print("\nX_selected.shape: {}".format(X_selected.shape))
# X_selected.shape: (100, 500)

# from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import Ridge
print("\nCross-validation accuracy (cv only on ridge): {:.2f}".format(
      np.mean(cross_val_score(Ridge(), X_selected, y, cv=5))))
# Cross-validation accuracy (cv only on ridge): 0.91

# The mean R^2 computed by cross-validation is 0.9, indicating a very good model. This can clearly not be right, as our data is entirely random. 
# What happened here is that our feature selection picked out some features among the 10000 random features that are (by chance) very well correlated with the target. 
# Because we fit the feature selection outside of the cross-validation, it could find features that are correlated both on the training and the test folds. 
# The information we leaked from the test-folds was very informative, leading to highly unrealistic results.

# Let’s compare this to a proper cross-validation using a pipeline:
pipe = Pipeline([("select", SelectPercentile(score_func=f_regression,
                                             percentile=5)),
                 ("ridge", Ridge())])
print("\nCross-validation accuracy (pipeline): {:.2f}".format(
      np.mean(cross_val_score(pipe, X, y, cv=5))))
# Cross-validation accuracy (pipeline): -0.25

# This time, we get a negative R^2 score, indicating a very poor model.
# Using the pipeline, the feature selection is now inside the cross-validation loop. 
# This means features can only be selected using the training folds of the data, not the test fold. 



## 6.4 The General Pipeline Interface
# The Pipeline class is not restricted to preprocessing and classification, but can in fact join any number of estimators together.
# For example, you could build a pipeline containing feature extraction(特征提取), feature selection(特征选择), scaling(缩放) and classification(分类), 
# for a total of four steps. Similarly the last step could be regression or clustering instead of classification.

# The only requirement for estimators in a pipeline is that all but the last step need to have a transform method, 
# so they can produce a new representation of the data that can be used in the next step.
print("\n----------- The General Pipeline Interface -----------")

def fit(self, X, y):
    X_transformed = X
    for name, estimator in self.steps[:-1]:
        # iterate over all but the final step
        # fit and transform the data
        X_transformed = estimator.fit_transform(X_transformed, y)
    # fit the last step
    self.steps[-1][1].fit(X_transformed, y)
    return self

def predict(self, X):
    X_transformed = X
    for step in self.steps[:-1]:
        # iterate over all but the final step
        # transform the data
        X_transformed = step[1].transform(X_transformed)
    # predict using the last step
    return self.steps[-1][1].predict(X_transformed)

## 6.4.1 Convenient Pipeline creation with make_pipeline
print("\n----------- Convenient Pipeline creation with make_pipeline -----------")

# from sklearn.pipeline import make_pipeline
# standard syntax
pipe_long = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC(C=100))])
# abbreviated syntax
pipe_short = make_pipeline(MinMaxScaler(), SVC(C=100))
print("\nPipeline steps(MinMaxScaler(), SVC(C=100)):\n{}".format(pipe_short.steps))
# Pipeline steps(MinMaxScaler(), SVC(C=100)):
# [('minmaxscaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('svc', SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
#     kernel='rbf', max_iter=-1, probability=False, random_state=None,
#     shrinking=True, tol=0.001, verbose=False))]

# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
pipe = make_pipeline(StandardScaler(), PCA(n_components=2), StandardScaler())
print("\nPipeline steps(StandardScaler(), PCA(n_components=2)):\n{}".format(pipe.steps))
# Pipeline steps(StandardScaler(), PCA(n_components=2)):
# [('standardscaler-1', StandardScaler(copy=True, with_mean=True, with_std=True)), ('pca', PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
#     svd_solver='auto', tol=0.0, whiten=False)), ('standardscaler-2', StandardScaler(copy=True, with_mean=True, with_std=True))]



## 6.4.2 Accessing step attributes
# Often you might want to inspect attributes of one of the steps of the pipeline, say the coefficients of a linear model or the components extracted by PCA. 
# The easiest way to access the step in a pipeline is the named_steps attribute, which is a dictionary from step names to the estimators:
print("\n----------- Accessing step attributes -----------")

# fit the pipeline defined before to the cancer dataset
pipe.fit(cancer.data)
# extract the first two principal components from the "pca" step
components = pipe.named_steps["pca"].components_
print("\ncomponents.shape: {}".format(components.shape))
# components.shape: (2, 30)



## 6.4.3 Accessing Attributes in a Pipeline inside GridSearchCV
# As we discussed above, one of the main reasons to use pipelines is for doing grid-searches. A common task then is to access some of the steps of a pipeline inside a grid-search.
# Let’s grid-search a LogisticRegression classifier on the cancer dataset, using Pipe line and StandardScaler to scale the data before passing it to the LogisticRegres sion classifier.
print("\n----------- Accessing Attributes in a Pipeline inside GridSearchCV -----------")

# from sklearn.linear_model import LogisticRegression

# First we create a pipeline using the make_pipeline function:
pipe = make_pipeline(StandardScaler(), LogisticRegression())

param_grid = {'logisticregression__C': [0.01, 0.1, 1, 10, 100]}
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=4)
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
print("\nGridSearchCV configuration:\n",grid)
# GridSearchCV configuration:
#  GridSearchCV(cv=5, error_score='raise-deprecating',
#              estimator=Pipeline(memory=None,
#                                 steps=[('standardscaler',
#                                         StandardScaler(copy=True,
#                                                        with_mean=True,
#                                                        with_std=True)),
#                                        ('logisticregression',
#                                         LogisticRegression(C=1.0,
#                                                            class_weight=None,
#                                                            dual=False,
#                                                            fit_intercept=True,
#                                                            intercept_scaling=1,
#                                                            l1_ratio=None,
#                                                            max_iter=100,
#                                                            multi_class='warn',
#                                                            n_jobs=None,
#                                                            penalty='l2',
#                                                            random_state=None,
#                                                            solver='warn',
#                                                            tol=0.0001,
#                                                            verbose=0,
#                                                            warm_start=False))],
#                                 verbose=False),
#              iid='warn', n_jobs=None,
#              param_grid={'logisticregression__C': [0.01, 0.1, 1, 10, 100]},
#              pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
#              scoring=None, verbose=0)


print("\nBest estimator:\n{}".format(grid.best_estimator_))
# Best estimator:
# Pipeline(memory=None,
#          steps=[('standardscaler',
#                  StandardScaler(copy=True, with_mean=True, with_std=True)),
#                 ('logisticregression',
#                  LogisticRegression(C=0.1, class_weight=None, dual=False,
#                                     fit_intercept=True, intercept_scaling=1,
#                                     l1_ratio=None, max_iter=100,
#                                     multi_class='warn', n_jobs=None,
#                                     penalty='l2', random_state=None,
#                                     solver='warn', tol=0.0001, verbose=0,
#                                     warm_start=False))],
#          verbose=False)


# This best_estimator_ in our case is a pipeline with two steps, "standardscaler" and "logisticregression". 
# To access the logisticregression step, we can use the named_steps attribute of the pipeline that we explained above:
print("\nLogistic regression step:\n{}".format(
      grid.best_estimator_.named_steps["logisticregression"]))
# Logistic regression step:
# LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
#                    intercept_scaling=1, l1_ratio=None, max_iter=100,
#                    multi_class='warn', n_jobs=None, penalty='l2',
#                    random_state=None, solver='warn', tol=0.0001, verbose=0,
#                    warm_start=False)


# Now that we have the trained LogisticRegression instance, we can access the coefficients (weights) associated with each input feature:
print("\nLogistic regression coefficients:\n{}".format(
      grid.best_estimator_.named_steps["logisticregression"].coef_))
# Logistic regression coefficients:
# [[-0.389 -0.375 -0.376 -0.396 -0.115  0.017 -0.355 -0.39  -0.058  0.209
#   -0.495 -0.004 -0.371 -0.383 -0.045  0.198  0.004 -0.049  0.21   0.224
#   -0.547 -0.525 -0.499 -0.515 -0.393 -0.123 -0.388 -0.417 -0.325 -0.139]]



## 6.5 Grid-searching preprocessing steps and model parameters
# Using pipelines, we can encapsulate all processing steps in our machine learning work flow in a single scikit-learn estimator. 
# Another benefit of doing this is that we can now adjust the parameters of the preprocessing using the outcome of a supervised task like regression or classification.

# In previous chapters, we used polynomial(多项式) features on the boston dataset before applying the ridge regressor. 
# Let’s model that using a pipeline. The pipeline contains three steps: scaling the data, computing polynomial features, and ridge regression:
print("\n----------- Grid-searching preprocessing steps and model parameters -----------")

# from sklearn.datasets import load_boston
boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target,
                                                    random_state=0)

# from sklearn.preprocessing import PolynomialFeatures
pipe = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(),
    Ridge())

# But how do we know which degree of polynomials to choose, or whether to choose any polynomials or interactions(交互项) at all? 
# Ideally we want to select the degree parameter based on the outcome of the classification.
# Using our pipeline, we can search over the degree parameter together with the parameter alpha of Ridge. 
# To do this, we define a param_grid that contains both, appropriately prefixed by the step names:
param_grid = {'polynomialfeatures__degree': [1, 2, 3],
              'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)
print("\nGridSearchCV configuration:\n", grid)
# GridSearchCV configuration:
#  GridSearchCV(cv=5, error_score='raise-deprecating',
#              estimator=Pipeline(memory=None,
#                                 steps=[('standardscaler',
#                                         StandardScaler(copy=True,
#                                                        with_mean=True,
#                                                        with_std=True)),
#                                        ('polynomialfeatures',
#                                         PolynomialFeatures(degree=2,
#                                                            include_bias=True,
#                                                            interaction_only=False,
#                                                            order='C')),
#                                        ('ridge',
#                                         Ridge(alpha=1.0, copy_X=True,
#                                               fit_intercept=True, max_iter=None,
#                                               normalize=False,
#                                               random_state=None, solver='auto',
#                                               tol=0.001))],
#                                 verbose=False),
#              iid='warn', n_jobs=-1,
#              param_grid={'polynomialfeatures__degree': [1, 2, 3],
#                          'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]},
#              pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
#              scoring=None, verbose=0)

print("\nBest parameters: {}".format(grid.best_params_))
print("\nTest-set score: {:.2f}".format(grid.score(X_test, y_test)))
# Best parameters: {'polynomialfeatures__degree': 2, 'ridge__alpha': 10}
# Test-set score: 0.77

# Let’s run a grid-search without polynomial features for comparison:
param_grid = {'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
pipe = make_pipeline(StandardScaler(), Ridge())
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
print("\nScore without poly features: {:.2f}".format(grid.score(X_test, y_test)))
# Score without poly features: 0.63

# As we had expected from the grid-search results visualized above, using no polyno‐ mial features leads to decidedly worse results. 



## 6.6 网格搜索选择使用哪个模型
pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC())])

# from sklearn.ensemble import RandomForestClassifier

param_grid = [
    {'classifier': [SVC()], 'preprocessing': [StandardScaler(), None],
     'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
     'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]},
    {'classifier': [RandomForestClassifier(n_estimators=100)],
     'preprocessing': [None], 'classifier__max_features': [1, 2, 3]}]

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)

print("\nBest params:\n{}\n".format(grid.best_params_))
print("\nBest cross-validation score: {:.2f}".format(grid.best_score_))
print("\nTest-set score: {:.2f}".format(grid.score(X_test, y_test)))



## 6.7 Summary and outlook
# In this chapter we introduced the Pipeline class a general purpose tool to chain together multiple processing steps in a machine learning work flow. 
# Real-world applications of machine learning are rarely an isolated use of a model, and instead a sequence of processing steps. 

# Using pipelines allows us to encapsulate multiple steps into a single python object that adheres to the familiar scikit-learn interface of fit, predict and transform.
# In particular when doing model evaluation using cross-validation and parameter selection using grid-search, using the Pipeline class to capture all processing steps is essential for proper evaluation.

# Choosing the right combination of feature extraction, preprocessing and models is somewhat of an art, that often requires some trial-and-error. 
# However, using pipelines, this “trying out” of many different processing steps is quite simple.


