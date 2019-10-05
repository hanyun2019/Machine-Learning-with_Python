# Introduction to Machine Learning with Python
# Chapter 3: Unsupervised Learning and Preprocessing
# Refreshed by Haowen Huang

import mglearn
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_blobs
from sklearn.datasets import fetch_lfw_people
from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.manifold import TSNE

## Unsupervised Learning and Preprocessing
# Unsupervised learning subsumes all kinds of machine learning where there is no known output, no teacher to instruct the learning algorithm. 
# In unsupervised learning, the learning algorithm is just shown the input data, and asked to extract knowledge from this data.

# Types of unsupervised learning
# 1) Unsupervised transformation (dimensionality reduction, “make up” the data)  无监督变换
# 2) Clustering algorithm (partition data into distinct groups of similar items) 聚类算法

# Challenges in unsupervised learning
# A major challenge in unsupervised learning is evaluating whether the algorithm learned something useful. 
# Often the only way to evaluate the result of an unsupervised algorithm is to inspect it manually.
# As a consequence, unsupervised algorithms are used often:
# 1) In an exploratory setting, when a data scientist wants to understand the data better, rather than as part of a larger automatic system. 
# 2) Another common application is as a preprocessing step for supervised algorithms. 
#    Learning a new representation of the data can sometimes improve the accuracy of supervised algorithms, or can lead to reduced memory and time consumption.

# Preprocessing and Scaling
# 1) StandardScaler: ensures that for each feature, the mean is zero, and the variance is one,this scaling does not ensure any particular minimum and maximum values for the features.
# 2) RobustScaler: uses the median and quartiles, instead of mean and variance.
# 3) MinMaxScaler: shifts the data such that all features are exactly between 0 and 1. 
# 4) Normalizer: it projects a data point on the circle, often used when only the direction (or angle) of the data matters, not the length of the feature vector.

# Applying data transformations
# As an example, say we want to apply the kernel SVM (SVC) to the cancer dataset, and use MinMaxScaler for preprocessing the data. 
# We start by loading and splitting our dataset into a training set and a test set. 
# We need a separate training and test set to evaluate the supervised model we will build after the preprocessing.
print("\n----------- Applying data transformations - breast_cancer dataset example -----------")

# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=1)
print("\nX_train.shape: ",X_train.shape)
print("\nX_test.shape: ",X_test.shape)
# X_train.shape:  (426, 30)
# X_test.shape:  (143, 30)

# from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)
print("\n scaler configuration:\n",scaler)

# transform data
X_train_scaled = scaler.transform(X_train)
# print dataset properties before and after scaling
print("\nX_train_scaled: transformed shape: {}".format(X_train_scaled.shape))
print("\nX_train: per-feature minimum before scaling:\n {}".format(X_train.min(axis=0)))
print("\nX_train: per-feature maximum before scaling:\n {}".format(X_train.max(axis=0)))
print("\nX_train_scaled: per-feature minimum after scaling:\n {}".format(X_train_scaled.min(axis=0)))
print("\nX_train_scaled: per-feature maximum after scaling:\n {}".format(X_train_scaled.max(axis=0)))

# transform test data
# To apply the SVM to the scaled data, we also need to transform the test set.
X_test_scaled = scaler.transform(X_test)
# print test data properties after scaling
print("\nX_test_scaled: per-feature minimum after scaling:\n{}".format(X_test_scaled.min(axis=0)))
print("\nX_test_scaled: per-feature maximum after scaling:\n{}".format(X_test_scaled.max(axis=0)))


# Scaling training and test data the same way
print("\n----------- Scaling training and test data the same way - example -----------")

# from sklearn.datasets import make_blobs

# make synthetic data
X, _ = make_blobs(n_samples=50, centers=5, random_state=4, cluster_std=2)
# split it into training and test set
X_train, X_test = train_test_split(X, random_state=5, test_size=.1)
# plot the training and test set
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].scatter(X_train[:, 0], X_train[:, 1], c='b', label="training set", s=60)
axes[0].scatter(X_test[:, 0], X_test[:, 1], marker='^', c='r', label="test set", s=60)
axes[0].legend(loc='upper left')
axes[0].set_title("original data")

# scale the data using MinMaxScaler
print("\n----------- MinMaxScaler -----------")
scaler = MinMaxScaler()
print("\n scaler configuration:\n",scaler)

scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
# visualize the properly scaled data
axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c='b', label="training set", s=60)
axes[1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], marker='^', c='r', label="test set", s=60)
axes[1].set_title("scaled data")

# rescale the test set separately, so that test set min is 0 and test set max is 1
# DO NOT DO THIS! For illustration purposes only
test_scaler = MinMaxScaler()
test_scaler.fit(X_test)
X_test_scaled_badly = test_scaler.transform(X_test)
# visualize wrongly scaled data
axes[2].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c='b', label="training set", s=60)
axes[2].scatter(X_test_scaled_badly[:, 0], X_test_scaled_badly[:, 1], marker='^', c='r', label="test set", s=60)
axes[2].set_title("improperly scaled data")

for ax in axes:
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
fig.tight_layout()

print("\n----------- StandardScaler -----------")

# from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
print("\n scaler configuration:\n",scaler)

# calling fit and transform in sequence (using method chaining)
X_scaled = scaler.fit(X_train).transform(X_train)
# same result, but more efficient computation
X_scaled_d = scaler.fit_transform(X_train)



## 3.3.4 The effect of preprocessing on supervised learning

print("\n----------- The effect of preprocessing on supervised learning: -----------")

# from sklearn.svm import SVC

# First, let’s fit the SVC on the original data again for comparison:
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
svm = SVC(C=100,gamma='auto')
svm.fit(X_train, y_train)
print("\nSVC(c=100) configuration:\n", svm)
print("\nTest set accuracy(using original data): {:.2f}".format(svm.score(X_test, y_test)))

# Result: 
# Test set accuracy: 0.63

# Now, let’s scale the data using MinMaxScaler before fitting the SVC:
# preprocessing using 0-1 scaling
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# learning an SVM on the scaled training data
svm.fit(X_train_scaled, y_train)

# scoring on the scaled test set
print("\nSVC(using MinMaxScaler()) configuration:\n", svm)
print("\nScaled test set accuracy(using MinMaxScaler()): {:.2f}".format(svm.score(X_test_scaled, y_test)))

# Result:
# Scaled test set accuracy: 0.97
# As we saw before, the effect of scaling the data is quite significant. 

# Change the scaler preprocessing algorithm from "MinMaxScaler()" to "StandardScaler()"
# preprocessing using zero mean and unit variance scaling

# from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# learning an SVM on the scaled training data
svm.fit(X_train_scaled, y_train)

# scoring on the scaled test set
print("\nSVC(using StandardScaler()) configuration:\n", svm)
print("\nSVM test accuracy(using StandardScaler()): {:.2f}".format(svm.score(X_test_scaled, y_test)))



## 3.4 Dimensionality Reduction, Feature Extraction and Manifold Learning  
##     降维、特征提取和流式学习

# As we discussed above, transforming data using unsupervised learning can have many motivations. 
# The most common motivations are visualization, compressing the data, and finding a representation that is more informative for further processing.

# Algorithms for the motivations:
# 1) PCA: Principal Component Analysis (the simplest and most widely used algorithms)  主成分分析
# 2) 


## 3.4.1 PCA: Principal Component Analysis
# 1. Applying PCA to the cancer dataset for visualization
# We can find the first two principal components, and visualize the data in this new, two-dimensional space, with a single scatter-plot.

print("\n----------- PCA: Principal Component Analysis -----------")
print("\n----------- 1. Applying PCA to the cancer dataset for visualization -----------")

# from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

# Before we apply PCA, we scale our data so that each feature has unit variance using StandardScaler:
scaler = StandardScaler()
scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)

# from sklearn.decomposition import PCA

# By default, PCA only rotates (and shifts) the data, but keeps all principal components. 
# To reduce the dimensionality of the data, we need to specify how many components we want to keep when creating the PCA object。

# keep the first two principal components of the data 保留数据的前两个成分
pca = PCA(n_components=2)
# fit PCA model to beast cancer data 对breast_cancer数据拟合PCA模型
pca.fit(X_scaled)

# transform data onto the first two principal components 将数据变换到前两个主成分的方向上
X_pca = pca.transform(X_scaled)
print("\npca.transform(X_scaled):\n",X_pca)
print("\nOriginal shape(X_scaled.shape): {}".format(str(X_scaled.shape)))
print("\nReduced shape(X_pca.shape): {}".format(str(X_pca.shape)))

# Result:
# Original shape(X_scaled.shape): (569, 30)
# Reduced shape(X_pca.shape): (569, 2)

# A downside of PCA is that the two axes in the plot above are often not very easy to interpret. 
# The principal components themselves are stored in the components_ attribute of the PCA during fitting.
print("\nPCA component shape: {}".format(pca.components_.shape))
# Result:
# PCA component shape: (2, 30)
print("\nPCA components:\n{}".format(pca.components_))


# 2. Eigenfaces for feature extraction 特征提取的特征脸
# Another application of PCA that we mentioned above is feature extraction. 
# The idea behind feature extraction is that it is possible to find a representation of your data 
# that is better suited to analysis than the raw representation you were given. 

print("\n----------- 2. Eigenfaces for feature extraction -----------")

# from sklearn.datasets import fetch_lfw_people

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

fig, axes = plt.subplots(2, 5, figsize=(15, 8),
                         subplot_kw={'xticks': (), 'yticks': ()})
for target, image, ax in zip(people.target, people.images, axes.ravel()):
    ax.imshow(image)
    ax.set_title(people.target_names[target])

print("\npeople.images.shape: {}".format(people.images.shape))
print("\nNumber of classes: {}".format(len(people.target_names)))

# Result:
# people.images.shape: (2211, 87, 65)

# Number of classes: 34
# Alejandro Toledo           39   Alvaro Uribe               35   Amelie Mauresmo            21   
# Andre Agassi               36   Ariel Sharon               77   Atal Bihari Vajpayee       24   
# Bill Clinton               29   Colin Powell              236   Donald Rumsfeld           121   
# George Robertson           22   George W Bush             530   Gerhard Schroeder         109   
# Gloria Macapagal Arroyo    44   Hamid Karzai               22   Hans Blix                  39   
# Igor Ivanov                20   Jacques Chirac             52   Jean Chretien              55   
# Jennifer Capriati          42   John Ashcroft              53   Juan Carlos Ferrero        28   
# Junichiro Koizumi          60   Kofi Annan                 32   Laura Bush                 41   
# Lleyton Hewitt             41   Megawati Sukarnoputri      33   Pete Sampras               22   
# Saddam Hussein             23   Serena Williams            52   Tiger Woods                23   
# Tom Daschle                25   Tony Blair                144   Vicente Fox                32   
# Vladimir Putin             49   

# count how often each target appears
counts = np.bincount(people.target)
# print counts next to target names:
for i, (count, name) in enumerate(zip(counts, people.target_names)):
    print("{0:25} {1:3}".format(name, count), end='   ')
    if (i + 1) % 3 == 0:
        print()

mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1
    
X_people = people.data[mask]
y_people = people.target[mask]

# scale the grey-scale values to be between 0 and 1
# instead of 0 and 255 for better numeric stability:
X_people = X_people / 255.

# A common task in face recognition is to ask if a previously unseen face belongs to a known person from a database. 
# One way to solve this problem would be to build a classifier where each person is a separate class. 
# A simple solution is to use a one-nearest-neighbor classifier which looks for the most similar face image to the face you are classifying. 
# A one-nearest-neighbor could in principle work with only a single training example per class. 

# from sklearn.neighbors import KNeighborsClassifier

# split the data in training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X_people, y_people, stratify=y_people, random_state=0)
# build a KNeighborsClassifier with using one neighbor:
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print("\nKNeighborsClassifier(n_neighbors=1):\n",knn)
print("\nTest set score of 1-nn((KNeighborsClassifier(n_neighbors=1))): {:.2f}".format(knn.score(X_test, y_test)))

# Result:
# Test set score of 1-nn: 0.36

# This is where PCA comes in. 
# Computing distances in the original pixel space is quite a bad way to measure similarity between faces. 
# We hope that using distances along principal components can improve our accuracy. 
# 
# Here we enable the whitening option of PCA, which rescales the principal components to have the same scale. 
# This is the same as using StandardScaler after the transformation. 

print("\n----------- PCA whitening -----------")

mglearn.plots.plot_pca_whitening()

# We fit the PCA object to the training data and extract the first 100 principal components. 
# Then we transform the training and test data:
pca = PCA(n_components=100, whiten=True, random_state=0).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print("\nX_train_pca.shape(PCA whitening): {}".format(X_train_pca.shape))
# Result:
# X_train_pca.shape(PCA whitening): (954, 100)

# The new data has 100 features, the first 100 principal components. 
# Now we can use the new representation to classify our images using one-nearest-neighbors:
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_pca, y_train)
print("\nKNeighborsClassifier(n_neighbors=1):\n",knn)
print("\nTest set accuracy(PCA whitening & KNeighborsClassifier(n_neighbors=1)): {:.2f}".format(knn.score(X_test_pca, y_test)))

# Result:
# Test set accuracy(PCA whitening & KNeighborsClassifier(n_neighbors=1)): 0.41
# Our accuracy improved quite significantly, from 36% to 41%, 
# confirming our intuition that the principal components might provide a better representation of the data.



## 3.4.2 NMF: Non-Negative Matrix Factorization 非负矩阵分解
# In PCA, we wanted components that are orthogonal, and that explain as much variance of the data as possible. 
# In NMF, we want the components and the coefficients to be non-negative; we want both the components and the coefficients to be greater or equal then zero.
# 
# The process of decomposing data into a non-negative weighted sum is particularly helpful for data that is created as the addition of several independent sources, 
# such as an audio track of multiple speakers, or music with many instruments.

print("\n----------- NMF: Non-Negative Matrix Factorization -----------")

mglearn.plots.plot_nmf_faces(X_train, X_test, image_shape)

# from sklearn.decomposition import NMF
nmf = NMF(n_components=15, random_state=0)
nmf.fit(X_train)
X_train_nmf = nmf.transform(X_train)
X_test_nmf = nmf.transform(X_test)

fig, axes = plt.subplots(3, 5, figsize=(15, 12),
                         subplot_kw={'xticks': (), 'yticks': ()})
for i, (component, ax) in enumerate(zip(nmf.components_, axes.ravel())):
    ax.imshow(component.reshape(image_shape))
    ax.set_title("{}. component".format(i))

compn = 3
# sort by 3rd component, plot first 10 images
inds = np.argsort(X_train_nmf[:, compn])[::-1]
fig, axes = plt.subplots(2, 5, figsize=(15, 8),
                         subplot_kw={'xticks': (), 'yticks': ()})
fig.suptitle("Large component 3")
for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
    ax.imshow(X_train[ind].reshape(image_shape))

compn = 7
# sort by 7th component, plot first 10 images
inds = np.argsort(X_train_nmf[:, compn])[::-1]
fig.suptitle("Large component 7")
fig, axes = plt.subplots(2, 5, figsize=(15, 8),
                         subplot_kw={'xticks': (), 'yticks': ()})
for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
    ax.imshow(X_train[ind].reshape(image_shape))

S = mglearn.datasets.make_signals()
plt.figure(figsize=(6, 1))
plt.plot(S, '-')
plt.xlabel("Time")
plt.ylabel("Signal")

# Mix data into a 100 dimensional state
A = np.random.RandomState(0).uniform(size=(100, 3))
X = np.dot(S, A.T)
print("\nShape of measurements: {}".format(X.shape))

nmf = NMF(n_components=3, random_state=42)
S_ = nmf.fit_transform(X)
print("\nRecovered signal shape: {}".format(S_.shape))

pca = PCA(n_components=3)
H = pca.fit_transform(X)

models = [X, S, S_, H]
names = ['Observations (first three measurements)',
         'True sources',
         'NMF recovered signals',
         'PCA recovered signals']

fig, axes = plt.subplots(4, figsize=(8, 4), gridspec_kw={'hspace': .5},
                         subplot_kw={'xticks': (), 'yticks': ()})

for model, name, ax in zip(models, names, axes):
    ax.set_title(name)
    ax.plot(model[:, :3], '-')

# Result:
# Shape of measurements: (2000, 100)
# Recovered signal shape: (2000, 3)
    
# There are many other algorithms that can be used to decompose each data point into a weighted sum of a fixed set of components, as PCA and NMF do. 
# Describing the constraints made on the components and coefficients often involves probability theory. 
# 
# If you are interested in these kinds of pattern extraction, we recommend to study the user guide of Independent Component Analysis (ICA), 
# Factor Analysis (FA) and Sparse Coding (dictionary learning), which are widely used decomposition methods. 
# http://scikit-learn.org/stable/modules/decomposition.html



# Manifold learning with t-SNE
# How to Use t-SNE Effectively: https://distill.pub/2016/misread-tsne/

print("\n----------- Manifold learning with t-SNE -----------")

# from sklearn.datasets import load_digits

digits = load_digits()

fig, axes = plt.subplots(2, 5, figsize=(10, 5),
                         subplot_kw={'xticks':(), 'yticks': ()})
for ax, img in zip(axes.ravel(), digits.images):
    ax.imshow(img)


# build a PCA model
pca = PCA(n_components=2)
pca.fit(digits.data)
# transform the digits data onto the first two principal components
digits_pca = pca.transform(digits.data)
colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525",
          "#A83683", "#4E655E", "#853541", "#3A3120", "#535D8E"]
plt.figure(figsize=(10, 10))
plt.xlim(digits_pca[:, 0].min(), digits_pca[:, 0].max())
plt.ylim(digits_pca[:, 1].min(), digits_pca[:, 1].max())
for i in range(len(digits.data)):
    # actually plot the digits as text instead of using scatter
    plt.text(digits_pca[i, 0], digits_pca[i, 1], str(digits.target[i]),
             color = colors[digits.target[i]],
             fontdict={'weight': 'bold', 'size': 9})
plt.xlabel("First principal component")
plt.ylabel("Second principal component")

# Result:
# Text(0, 0.5, 'Second principal component')


# from sklearn.manifold import TSNE

tsne = TSNE(random_state=42)
print("\nTSNE(random_state=42):\n",tsne)
# use fit_transform instead of fit, as TSNE has no transform method
digits_tsne = tsne.fit_transform(digits.data)

plt.figure(figsize=(10, 10))
plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max() + 1)
plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max() + 1)
for i in range(len(digits.data)):
    # actually plot the digits as text instead of using scatter
    plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]),
             color = colors[digits.target[i]],
             fontdict={'weight': 'bold', 'size': 9})
plt.xlabel("t-SNE feature 0")
plt.ylabel("t-SNE feature 1")

# Result:
# Text(0, 0.5, 't-SNE feature 1')











