# Introduction to Machine Learning with Python
# Chapter 2: Supervised Learning
# Refreshed by Haowen Huang

import mglearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

### Supervised Learning
## 2.1 Classification and Regression
# An easy way to distinguish between classification and regression tasks is to ask whether there is some kind of ordering 
# or continuity in the output. 
# If there is an ordering, or a continuity between possible outcomes, then the problem is a regression problem.

## 2.2 Generalization, Overfitting, and Underfitting

## 2.3 Supervised Machine Learning Algorithms
# 2.3.1 Some Sample Datasets

# An example of a synthetic two-class classification dataset is the forge dataset, which has two features. 
# Below is a scatter plot visualizing all of the data points in this dataset. 
# The plot has the first feature on the x-axis and the second feature on the y-axis. 
# As is always the case in in scatter plots, each data point is represented as one dot. 
# The color of the dot indicates its class, with red meaning class 0 and blue meaning class 1.


print("\n----------- Classification dataset example -----------")
# mglearn.plots.plot_grid_search_overview()

# generate dataset
X, y = mglearn.datasets.make_forge()
# plot dataset
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)   # X[:, 0]      取所有行的第0个数据
                                                # X[:, 1]      取所有行的第1个数据    
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("\nX.shape:", X.shape)
print("\nX", X)
print("\ny.shape:", y.shape)
print("\ny", y)

# X.shape: (26, 2)
# As you can see from X.shape, this dataset consists of 26 data points, with 2 features.


# To illustrate regression algorithms, we will use the synthetic wave dataset shown below. 
# The wave dataset only has a single input feature, and a continuous target variable (or response) that we want to model.
# The plot below is showing the single feature on the x-axis, with the data points as green dots. 
# For each data point, the target output is plotted in blue on the y-axis.

print("\n----------- Regression dataset example -----------")
 
X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("Feature")
plt.ylabel("Target")

print("\nX.shape:", X.shape)
print("\nX", X)
print("\ny.shape:", y.shape)
print("\ny", y)


# We will complement these small synthetic dataset with two real-world datasets that are included in scikit-learn. 
# One is the Wisconsin breast cancer dataset (or cancer for short), which records clinical measurements of breast cancer tumors. 
# Each tumor is labeled as “benign” (for harmless tumors) or “malignant” (for cancerous tumors), 
# and the task is to learn to predict whether a tumor is malignant based on the measure‐ ments of the tissue.

# The data can be loaded using the load_breast_cancer from scikit-learn. Datasets that are included in scikit-learn are usually 
# stored as Bunch objects, which contain some information about the dataset as well as the actual data.
# All you need to know about Bunch objects is that they behave like dictionaries, 
# with the added benefit that you can access values using a dot (as in bunch.key instead of bunch['key']).

print("\n----------- Real-world classification datasets example: the Wisconsin breast cancer dataset -----------")

# from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("\nBreast cancer dataset cancer.keys():\n", cancer.keys())

print("\nBreast cancer dataset shape of cancer data:\n", cancer.data.shape)

print("\nBreast cancer dataset sample counts per class:\n",
      {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))})

print("\nBreast cancer dataset feature names:\n", cancer.feature_names)
# print("\nDescriptions:\n",cancer.DESCR)


# We will also be using a real-world regression dataset, the Boston Housing dataset. 
# The task associated with this dataset is to predict the median value of homes in several Boston neighborhoods in the 1970s, 
# using information about the neighborhoods such as crime rate, proximity to the Charles River, highway accessibility and so on.

print("\n----------- Real-world regression datasets example: the Boston Housing dataset -----------")

# from sklearn.datasets import load_boston
boston = load_boston()
print("\nBoston housing keys:",boston.keys())
# print("\nBoston housing basic features:",boston.DESCR)

print("\nBoston housing data shape:", boston.data.shape)
print("\nBoston housing data:",boston.data)

print("\nBoston housing data feature names:",boston.feature_names)
print("\nBoston housing data[0]:",boston.data[0])


# Boston housing data shape: (506, 13)

# Feature Engineering(特征工程)：将特征的乘积也做为特征
# For our purposes here, we will actually expand this dataset, by not only considering these 13 measurements as input features, 
# but also looking at all products (also called interactions) between features.
# In other words, we will not only consider crime rate and highway accessibility as a feature, 
# but also the product of crime rate and highway accessibility. 
# Including derived feature like these is called feature engineering, which we will discuss in more detail in Chapter 5 (Representing Data).
# This derived dataset can be loaded using the load_extended_boston function:

X, y = mglearn.datasets.load_extended_boston()
print("\nBoston housing extended data(features producted as new feature) X.shape:", X.shape)

# Boston housing extended data(features producted as new feature) X.shape: (506, 104)
# The resulting 104 features are the 13 original features, the 13 choose 2 = 91 (Footnote: the number of ways to pick 2 elements out of 13 elements) features 
# that are product of two features, and one constant feature.
# 第1个特征可以和13个特征相乘，第2个特征可以和12个特征相乘(除了第1个)，第3个特征可以和11个特征相乘......依次相加：13+12+11+...+1=91
# 91(相乘出来的新特征) + 13(原有的13个特征) = 104



# 2.3.2 KNN: k-Nearest Neighbor
# 2.3.2.1 k-Neighbors classification
# When considering more than one neighbor, we use voting to assign a label. 
# This means, for each test point, we count how many neighbors are red, and how many neighbors are blue. 
# We then assign the class that is more frequent: in other words, the majority class among the k neighbors.
# Below is an illustration using the three closest neighbors.

print("\n----------- KNN(k-Nearest Neighbor) Algorithm: classification example -----------")

# from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_forge()
print("\nOriginal dataset X:\n", X)
print("\nOriginal dataset X.shape:\n", X.shape)
print("\nOriginal dataset y:\n", y)
print("\nOriginal dataset y.shape:\n", y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print("\nDataset X_train:\n", X_train)
print("\nDataset X_test:\n", X_test)
print("\nDataset y_train:\n", y_train)
print("\nDataset y_test:\n", y_test)

# from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)

# Now we fit the classifier using the training set. 
# For KNeighborsClassifier this means storing the dataset, so we can compute neighbors during prediction.
clf.fit(X_train, y_train)
print("\nKNeighborsClassifier configuration:\n",clf)

# To make predictions on the test data, we call the predict method. 
# This computes the nearest neighbors in the training set and finds the most common class among these:
print("\nKNN example test set predictions: ", clf.predict(X_test))
print("\nKNN example test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))


# 2.3.2.2 Analyzing KNeighborsClassifier
# For two-dimensional datasets, we can also illustrate the prediction for all possible test point in the xy-plane. 
# We color the plane red in regions where points would be assigned the red class, and blue otherwise. 
# This lets us view the decision boundary, which is the divide between where the algorithm assigns class red versus 
# where it assigns class blue.
# Here is a visualization of the decision boundary for one, three and five neighbors:

fig, axes = plt.subplots(1, 3, figsize=(10, 3))

for n_neighbors, ax in zip([1, 3, 9], axes):
    # the fit method returns the object self, so we can instantiate and fit in one line
    # fit方法返回对象本身，所以我们可以将实例化和拟合放在一行代码中
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} neighbor(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=3)
 
# Considering more and more neighbors leads to a smoother decision boundary. 
# A smoother boundary corresponds to a simple model. 
# In other words, using few neighbors corresponds to high model complexity, 
# and using many neighbors corresponds to low model complexity.



# Using KNN algorithm to investigate the real world breast cancer dataset
# To investigate whether we can confirm the connection between model complexity and generalization.

print("\n----------- KNN(k-Nearest Neighbor) Algorithm: breast cancer dataset classification example -----------")

# from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=66)

training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))
    
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()

# The test set accuracy for using a single neighbor is lower then when using more neighbors, 
# indicating that using a single nearest neighbor leads to a model that is too complex. 
# On the other hand, when considering 10 neighbors, the model is too simple, and performance is even worse. 
# The best performance is somewhere in the middle, around using six neighbors.

# Still, the worst performance is around 88% accuracy, which might still be acceptable.



# 2.3.2.3 k-Neighbors Regression
# There is also a regression variant of the k-nearest neighbors algorithm, this time using the wave dataset. 
# The k nearest neighbors algorithm for regression is implemented in the KNeighbors Regressor class in scikit-learn.
print("\n----------- KNN(k-Nearest Neighbor) Algorithm: wave dataset regression example -----------")

# from sklearn.neighbors import KNeighborsRegressor

X, y = mglearn.datasets.make_wave(n_samples=40)

# split the wave dataset into a training and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# instantiate the model and set the number of neighbors to consider to 3
reg = KNeighborsRegressor(n_neighbors=3)
# fit the model using the training data and training targets
reg.fit(X_train, y_train)
print("\nKNN wave dataset regression example - KNeighborsRegressor configuration:\n", reg)

print("\nKNN wave dataset regression example - y_test:\n", y_test)
# Now we can make predictions on the test set
print("\nKNN wave dataset regression example - test set predictions:\n", reg.predict(X_test))

# The $R^2$ score, also known as coefficient of determination, 
# is a measure of goodness of a prediction for a regression model, and yields a score up to 1. 
# A value of 1 corresponds to a perfect prediction, and a value of 0 corresponds to a constant model 
# that just predicts the mean of the training set responses y_train.
print("\nKNN wave dataset regression example - test set R^2: {:.2f}\n".format(reg.score(X_test, y_test)))



# 2.3.2.4 Analyzing k nearest neighbors regression
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# create 1,000 data points, evenly spaced between -3 and 3
# 创建1000个数据点，在-3和3之间均匀分布
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
    # make predictions using 1, 3, or 9 neighbors
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
    ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)

    ax.set_title(
        "{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(
            n_neighbors, reg.score(X_train, y_train),
            reg.score(X_test, y_test)))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
axes[0].legend(["Model predictions", "Training data/target",
                "Test data/target"], loc="best")

# Using only a single neighbor, each point in the training set has an obvious influence on the predictions, 
# and the predicted values go through all of the data points. This leads to a very unsteady prediction. 
# Considering more neighbors leads to smoother predictions, but these do not fit the training data as well.



# 2.3.2.5 Strengths, weaknesses, and parameters
# In principal, there are two important parameters to the KNeighbors classifier: 
# 1）the number of neighbors； 2）how you measure distance between data points. 
# For 2): by default, Euclidean distance(欧式距离) is used, which works well in many settings.

# So while the nearest neighbors algorithm is easy to understand, it is not often used in practice, 
# due to prediction being slow, and its inability to handle many features. 



## 2.4 Linear models
# 2.4.1 Linear models for regression



















