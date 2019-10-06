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
from sklearn.datasets import make_moons

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import silhouette_score

from scipy.cluster.hierarchy import dendrogram, ward

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



## 3.5 Clustering   聚类
## 3.5.1 k-Means clustering     k均值聚类
# The algorithm alternates between two steps: 
# 1) assigning each data point to the closest cluster center, 
# 2) setting each cluster center as the mean of the data points that are assigned to it.

print("\n----------- Clustering - k-Means clustering  -----------")

# We specified that we are looking for three clusters, so the algorithm was initialized by declaring three data points as cluster centers (see “Initialization”). 
# Then the iterative algorithm starts: Each data point is assigned to the cluster center it is closest to (see “Assign Points (1)”). 
# Next, the cluster centers are updated to be the mean of the assigned points (see “Recompute Centers (1)”). 
# Then the process is repeated. After the second iteration, the assignment of points to cluster centers remained unchanged, so the algorithm stops.
# Given new data points, k-Means will assign them to the closest cluster center. 
mglearn.plots.plot_kmeans_algorithm()
mglearn.plots.plot_kmeans_boundaries()

# from sklearn.datasets import make_blobs
# from sklearn.cluster import KMeans

# generate synthetic two-dimensional data
X, y = make_blobs(random_state=1)

# build the clustering model
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
print("\nKMeans(n_clusters=3):\n", kmeans)
print("\nCluster memberships:\n{}".format(kmeans.labels_))
print("\nkmeans.predict(X):\n",kmeans.predict(X))


# Vector Quantization(矢量量化) - Or Seeing k-Means as Decomposition(将k均值看作分解)
# k-Means tries to represent each data point using a cluster center. You can think of that as each point being represented using only a single component, 
# which is given by the cluster center. This view of k-Means as a decomposition method, where each point is represented using a single component, 
# is called vector quantization.   

# Here is a side-by-side comparison of PCA, NMF and k-Means, showing the components extracted, as well as reconstructions of faces from the test set using 100 components. 
# For k-Means, the reconstruction is the closest cluster center found on the training set:
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)
nmf = NMF(n_components=100, random_state=0)
nmf.fit(X_train)
pca = PCA(n_components=100, random_state=0)
pca.fit(X_train)
kmeans = KMeans(n_clusters=100, random_state=0)
kmeans.fit(X_train)

X_reconstructed_pca = pca.inverse_transform(pca.transform(X_test))
X_reconstructed_kmeans = kmeans.cluster_centers_[kmeans.predict(X_test)]
X_reconstructed_nmf = np.dot(nmf.transform(X_test), nmf.components_)

fig, axes = plt.subplots(3, 5, figsize=(8, 8), subplot_kw={'xticks': (), 'yticks': ()})
fig.suptitle("Extracted Components")
for ax, comp_kmeans, comp_pca, comp_nmf in zip(
        axes.T, kmeans.cluster_centers_, pca.components_, nmf.components_):
    ax[0].imshow(comp_kmeans.reshape(image_shape))
    ax[1].imshow(comp_pca.reshape(image_shape), cmap='viridis')
    ax[2].imshow(comp_nmf.reshape(image_shape))

axes[0, 0].set_ylabel("kmeans")
axes[1, 0].set_ylabel("pca")
axes[2, 0].set_ylabel("nmf")

fig, axes = plt.subplots(4, 5, subplot_kw={'xticks': (), 'yticks': ()},
                         figsize=(8, 8))
fig.suptitle("Reconstructions")
for ax, orig, rec_kmeans, rec_pca, rec_nmf in zip(
        axes.T, X_test, X_reconstructed_kmeans, X_reconstructed_pca,
        X_reconstructed_nmf):

    ax[0].imshow(orig.reshape(image_shape))
    ax[1].imshow(rec_kmeans.reshape(image_shape))
    ax[2].imshow(rec_pca.reshape(image_shape))
    ax[3].imshow(rec_nmf.reshape(image_shape))

axes[0, 0].set_ylabel("original")
axes[1, 0].set_ylabel("kmeans")
axes[2, 0].set_ylabel("pca")
axes[3, 0].set_ylabel("nmf")


# An interesting aspect of vector quantization using k-Means is that we can use many more clusters than input dimensions to encode our data. 
# Let’s go back to the two_moons data. Using PCA or NMF, there is nothing much we can do to this data, as it lives in only two dimensions. 
# Reducing it to one dimension with PCA or NMF would completely destroy the structure of the data. 
# But we can find a more expressive representation using k-Means, by using more cluster centers:

print("\n----------- Clustering - k-Means clustering - two_moons dataset example -----------")

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(X)
y_pred = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=60, cmap='Paired')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=60,
            marker='^', c=range(kmeans.n_clusters), linewidth=2, cmap='Paired')
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
print("\nK-Means clustering - two_moons dataset - Cluster memberships:\n{}".format(y_pred))

distance_features = kmeans.transform(X)
print("\nK-Means clustering - two_moons dataset - Distance feature shape: {}".format(distance_features.shape))
print("\nK-Means clustering - two_moons dataset - Distance features:\n{}".format(distance_features))

# k-Means is a very popular algorithm for clustering, not only because it is relatively easy to understand and implement, but also because it runs relatively quickly. 
# Drawbacks of k-Means:
# 1) It relies on a random initialization, which means the outcome of the algorithm depends on a random seed. By default, scikit-learn runs the algorithm 10 times with 10 different random initializations, and returns the best result.
# 2) The relatively restrictive assumptions made on the shape of clusters, and the requirement to specify the number of clusters you are looking for (which might not be known in a real-world application).



## 3.5.2 Agglomerative Clustering(凝聚聚类)
# Agglomerative clustering refers to a collection of clustering algorithms that all build upon the same principles: 
# The algorithm starts by declaring each point its own cluster, and then merges the two most similar clusters until some stopping criterion is satisfied.

print("\n----------- Clustering - Agglomerative Clustering -----------")

# from sklearn.cluster import AgglomerativeClustering

X, y = make_blobs(random_state=1)

agg = AgglomerativeClustering(n_clusters=3)
assignment = agg.fit_predict(X)

mglearn.discrete_scatter(X[:, 0], X[:, 1], assignment)
plt.legend(["Cluster 0", "Cluster 1", "Cluster 2"], loc="best")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")


# Hierarchical Clustering and Dendrograms(层次聚类与树状图)
# While this visualization provides a very detailed view of the hierarchical clustering, it relies on the two-dimensional nature of the data, 
# and can therefore not be used on datasets that have more than two features. 
# There is, however, another tool to visualize hierarchical clustering, called a dendrogram.

print("\n----------- Clustering - Agglomerative Clustering - Hierarchical Clustering and Dendrograms -----------")
# Import the dendrogram function and the ward clustering function from SciPy

# from scipy.cluster.hierarchy import dendrogram, ward

X, y = make_blobs(random_state=0, n_samples=12)
# Apply the ward clustering to the data array X
# The SciPy ward function returns an array that specifies the distances
# bridged when performing agglomerative clustering
linkage_array = ward(X)
# Now we plot the dendrogram for the linkage_array containing the distances
# between clusters
dendrogram(linkage_array)

# mark the cuts in the tree that signify two or three clusters
ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [7.25, 7.25], '--', c='k')
ax.plot(bounds, [4, 4], '--', c='k')

ax.text(bounds[1], 7.25, ' two clusters', va='center', fontdict={'size': 15})
ax.text(bounds[1], 4, ' three clusters', va='center', fontdict={'size': 15})
plt.xlabel("Sample index")
plt.ylabel("Cluster distance")

# Unfortunately, agglomerative clustering still fails at separating complex shapes like the two_moons dataset. 
# The same is not true for the next algorithm we will look at, DBSCAN.


## 3.5.3 DBSCAN (Density-based spatial clustering of applications with noise, 具有噪声的基于密度的空间聚类应用)
#  The main benefits of DBSCAN are that：
#  a) it does not require the user to set the number of clusters a priori, 
#  b) it can capture clusters of complex shapes, 
#  c) it can identify point that are not part of any cluster.

# DBSCAN is somewhat slower than agglomerative clustering and k-Means, but still scales to relatively large datasets.

# The way DBSCAN works is by identifying points that are in “crowded” regions of the feature space, where many data points are close together. 
# These regions are referred to as dense regions in feature space. 
# The idea behind DBSCAN is that clusters form dense regions of data, separated by regions that are relatively empty.

print("\n----------- Clustering - DBSCAN  -----------")

# from sklearn.cluster import DBSCAN

X, y = make_blobs(random_state=0, n_samples=12)

dbscan = DBSCAN()
clusters = dbscan.fit_predict(X)
print("\nClustering - DBSCAN - Cluster memberships:\n{}".format(clusters))

# Result:
# Clustering - DBSCAN - Cluster memberships:
# [-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]

# As you can see, all data points were assigned the label -1, which stands for noise. 
# This is a consequence of the default parameter settings for eps and min_samples, which are not attuned to small toy datasets.

# Let’s investigate the effect of changing eps and min_samples.
mglearn.plots.plot_dbscan()

# Result:
# min_samples: 2 eps: 1.000000  cluster: [-1  0  0 -1  0 -1  1  1  0  1 -1 -1]
# min_samples: 2 eps: 1.500000  cluster: [0 1 1 1 1 0 2 2 1 2 2 0]
# min_samples: 2 eps: 2.000000  cluster: [0 1 1 1 1 0 0 0 1 0 0 0]
# min_samples: 2 eps: 3.000000  cluster: [0 0 0 0 0 0 0 0 0 0 0 0]
# min_samples: 3 eps: 1.000000  cluster: [-1  0  0 -1  0 -1  1  1  0  1 -1 -1]
# min_samples: 3 eps: 1.500000  cluster: [0 1 1 1 1 0 2 2 1 2 2 0]
# min_samples: 3 eps: 2.000000  cluster: [0 1 1 1 1 0 0 0 1 0 0 0]
# min_samples: 3 eps: 3.000000  cluster: [0 0 0 0 0 0 0 0 0 0 0 0]
# min_samples: 5 eps: 1.000000  cluster: [-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]
# min_samples: 5 eps: 1.500000  cluster: [-1  0  0  0  0 -1 -1 -1  0 -1 -1 -1]
# min_samples: 5 eps: 2.000000  cluster: [-1  0  0  0  0 -1 -1 -1  0 -1 -1 -1]
# min_samples: 5 eps: 3.000000  cluster: [0 0 0 0 0 0 0 0 0 0 0 0]


# Below is the result of DBSCAN running on the two_moons dataset. 
# The algorithm actually finds the two half-circles and separates them using the default settings.
print("\n----------- Clustering - DBSCAN - two_moons dataset example -----------")

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

# Rescale the data to zero mean and unit variance
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

dbscan = DBSCAN()
clusters = dbscan.fit_predict(X_scaled)
# plot the cluster assignments
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm2, s=60)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")



## 3.5.4 Comparing and evaluating clustering algorithms
# 1) Evaluating clustering with ground truth 用真实值评估聚类
# 2) Evaluating clustering without ground truth 在没有真实值的情况下评估聚类
# 3) Comparing algorithms on the faces dataset 用人脸数据集评估聚类

print("\n----------- Clustering - Comparing and evaluating clustering algorithms -----------")
print("\n----------- Evaluating clustering with ground truth -----------")

# from sklearn.metrics.cluster import adjusted_rand_score

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

# Rescale the data to zero mean and unit variance
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

fig, axes = plt.subplots(1, 4, figsize=(15, 3),
                         subplot_kw={'xticks': (), 'yticks': ()})

# make a list of algorithms to use
algorithms = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2),
              DBSCAN()]

# create a random cluster assignment for reference
random_state = np.random.RandomState(seed=0)
random_clusters = random_state.randint(low=0, high=2, size=len(X))

# plot random assignment
axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_clusters,
                cmap=mglearn.cm3, s=60)
axes[0].set_title("Random assignment - ARI: {:.2f}".format(
        adjusted_rand_score(y, random_clusters)))

for ax, algorithm in zip(axes[1:], algorithms):
    # plot the cluster assignments and cluster centers
    clusters = algorithm.fit_predict(X_scaled)
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters,
               cmap=mglearn.cm3, s=60)
    ax.set_title("{} - ARI: {:.2f}".format(algorithm.__class__.__name__,
                                           adjusted_rand_score(y, clusters)))


# The problem in using accuracy is that it requires the assigned cluster labels to exactly match the ground truth. 
# However, the cluster labels themselves are meaningless, and only which points are in the same cluster matters:

# from sklearn.metrics import accuracy_score

# these two labelings of points correspond to the same clustering
clusters1 = [0, 0, 1, 1, 0]
clusters2 = [1, 1, 0, 0, 1]
# accuracy is zero, as none of the labels are the same
print("\nEvaluating clustering with ground truth - Accuracy: {:.2f}".format(accuracy_score(clusters1, clusters2)))
# adjusted rand score is 1, as the clustering is exactly the same:
print("\nEvaluating clustering with ground truth - ARI: {:.2f}".format(adjusted_rand_score(clusters1, clusters2)))


print("\n----------- Evaluating clustering without ground truth -----------")

# from sklearn.metrics.cluster import silhouette_score

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

# rescale the data to zero mean and unit variance
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

fig, axes = plt.subplots(1, 4, figsize=(15, 3),
                         subplot_kw={'xticks': (), 'yticks': ()})

# create a random cluster assignment for reference
random_state = np.random.RandomState(seed=0)
random_clusters = random_state.randint(low=0, high=2, size=len(X))

# plot random assignment
axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_clusters,
                cmap=mglearn.cm3, s=60)
axes[0].set_title("Random assignment: {:.2f}".format(
    silhouette_score(X_scaled, random_clusters)))

algorithms = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2),
              DBSCAN()]

for ax, algorithm in zip(axes[1:], algorithms):
    clusters = algorithm.fit_predict(X_scaled)
    # plot the cluster assignments and cluster centers
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm3,
               s=60)
    ax.set_title("{} : {:.2f}".format(algorithm.__class__.__name__,
                                      silhouette_score(X_scaled, clusters)))

# As you can see, k-Means gets the highest silhouette score, even though we might prefer the result produced by DBSCAN.
# Only if we actually analyze the clusters can we know whether the clustering corre‐ sponds to anything we are interested in.



# Comparing algorithms on the faces dataset
# Let’s apply the k-Means, DBSCAN and agglomerative clustering algorithms to the labeled faces in the wild dataset, 
# and see if any of them find interesting structure.
print("\n----------- Comparing k-Means, DBSCAN and agglomerative clustering algorithms on the labeled faces dataset -----------")

# extract eigenfaces from lfw(labeled faces wild dataset) data and transform data
# We will use the eigenface representation of the data, as produced by PCA(whiten=True), with 100 components. 
# We saw above that this is a more semantic representation of the face images than the raw pixels. It will also make computation faster.

# from sklearn.decomposition import PCA
pca = PCA(n_components=100, whiten=True, random_state=0)
X_pca = pca.fit_transform(X_people)

print("\n----------- Analyzing the labeled faces dataset with DBSCAN-----------")

# Analyzing the faces dataset with DBSCAN
# apply DBSCAN with default parameters
dbscan = DBSCAN()
labels = dbscan.fit_predict(X_pca)
print("\nAnalyzing the faces dataset with DBSCAN - Unique labels: {}".format(np.unique(labels)))
# Result:
# Analyzing the faces dataset with DBSCAN - Unique labels: [-1]

# We see that all returned labels are -1, so all of the data was labeled as “noise” by DBSCAN. 
# There are two things we can change to help this: 
# a) we can make eps higher, to expand the neighborhood of each point
# b) we can make min_samples lower, to consider smaller groups of points as clusters.

# Let’s try changing min_samples first
dbscan = DBSCAN(min_samples=3)
labels = dbscan.fit_predict(X_pca)
print("\nAnalyzing the faces dataset with DBSCAN(min_samples=3) - Unique labels: {}".format(np.unique(labels)))
# Result:
# Analyzing the faces dataset with DBSCAN(min_samples=3) - Unique labels: [-1]

# Even when considering groups of three points, everything is labeled as noise. 
# So we need to increase eps.

dbscan = DBSCAN(min_samples=3, eps=15)
labels = dbscan.fit_predict(X_pca)
print("\nAnalyzing the faces dataset with DBSCAN(min_samples=3, eps=15) - Unique labels: {}".format(np.unique(labels)))
# Result:
# Analyzing the faces dataset with DBSCAN(min_samples=3, eps=15) - Unique labels: [-1  0]

# Using a much larger eps=15 we get only a single clusters and noise points. 
# We can use this result and find out what the “noise” looks like compared to the rest of the data. 
# To understand better what’s happening, let’s look at how many points are noise, and how many points are inside the cluster:

# count number of points in all clusters and noise.
# bincount doesn't allow negative numbers, so we need to add 1.
# the first number in the result corresponds to noise points
print("Number of points per cluster: {}".format(np.bincount(labels + 1)))
# Result:
# Number of points per cluster: [  25 1247]

# There are only very few noise points, 27, so we can look at all of them:
noise = X_people[labels==-1]

fig, axes = plt.subplots(3, 9, subplot_kw={'xticks': (), 'yticks': ()}, figsize=(12, 4))
for image, ax in zip(noise, axes.ravel()):
    ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)

# outlier detection:
# Comparing these images to the random sample of face images from Figure some_faces, we can guess why they were labeled as noise: 
# the image in the sixth image in the first row one has a person drinking from a glass, 
# there are images with hats, 
# and the second to last image has a hand in front of the face. 
# The other images contain odd angles or crops that are too close or too wide.
# This kind of analysis, trying to find “the odd one out”, is called outlier detection. 

# If we want to find more interesting clusters than just one large one, we need to set eps smaller, somewhere between 15 and 0.5 (the default). 
# Let’s have a look at what different values of eps result in:

for eps in [1, 3, 5, 7, 9, 11, 13]:
    print("\neps={}".format(eps))
    dbscan = DBSCAN(eps=eps, min_samples=3)
    labels = dbscan.fit_predict(X_pca)
    print("Number of clusters: {}".format(len(np.unique(labels))))
    print("Cluster sizes: {}".format(np.bincount(labels + 1)))
# Result:
# eps=1
# Number of clusters: 1
# Cluster sizes: [1272]

# eps=3
# Number of clusters: 1
# Cluster sizes: [1272]

# eps=5
# Number of clusters: 1
# Cluster sizes: [1272]

# eps=7
# Number of clusters: 9
# Cluster sizes: [1228    3   20    6    3    3    3    3    3]

# eps=9
# Number of clusters: 3
# Cluster sizes: [810 458   4]

# eps=11
# Number of clusters: 2
# Cluster sizes: [284 988]

# eps=13
# Number of clusters: 2
# Cluster sizes: [  91 1181]


# For small numbers of eps, again all points are labeled as noise. 
# For eps=7, we get many noise points, and many smaller clusters. 
# For eps=9 we still get many noise points, one big cluster and some smaller clusters. 
# Starting from eps=11 we get only one large cluster and noise.
# What is interesting to note is that there are never more than one large cluster. 


# Analyzing the faces dataset with k-Means
print("\n----------- Analyzing the labeled faces dataset with k-Means -----------")

# extract clusters with k-Means
km = KMeans(n_clusters=10, random_state=0)
labels_km = km.fit_predict(X_pca)
print("\nCluster sizes k-means: {}".format(np.bincount(labels_km)))
# Result:
# Cluster sizes k-means: [ 90 352 107  86  93 169  29 132 121  93]

#As you can see, k-Means clustering partitioned the data into relatively similarly sized clusters from 86 to 352. 
# This is quite different from the result of DBSCAN.

fig, axes = plt.subplots(2, 5, subplot_kw={'xticks': (), 'yticks': ()},
                         figsize=(12, 4))
for center, ax in zip(km.cluster_centers_, axes.ravel()):
    ax.imshow(pca.inverse_transform(center).reshape(image_shape),
              vmin=0, vmax=1)

# The cluster centers found by k-Means are very smooth version of faces. 
# This is not very surprising, given that each center is an average of 86 to 352 face images. 


# Analyzing the faces dataset with agglomerative clustering
print("\n----------- Analyzing the faces dataset with agglomerative clustering -----------")

# extract clusters with ward agglomerative clustering
agglomerative = AgglomerativeClustering(n_clusters=10)
labels_agg = agglomerative.fit_predict(X_pca)
print("\ncluster sizes agglomerative clustering: {}".format(np.bincount(labels_agg)))
# Result:
# cluster sizes agglomerative clustering: [535 237 126 182  37   4  56  72  15   8]

# Agglomerative clustering produces relatively equally sized clusters, with cluster sizes between 4 and 535. 
# These are more uneven than k-Means, but much more even than the ones produced by DBSCAN.

# We can compute the ARI to measure if the two partitions of the data given by agglomerative clustering and k-Means are similar:
print("\nARI: {:.2f}".format(adjusted_rand_score(labels_agg, labels_km)))
# Result:
# ARI: 0.09

# An ARI of only 0.09 means that the two clusterings labels_agg and labels_km have quite little in common. 
# This is not very surprising, given the fact that points further away from the cluster centers seem to have little in common for k-Means.



## 3.6 Summary and Outlook
# This chapter introduced a range of unsupervised learning algorithms that can be applied for exploratory data analysis and preprocessing. 
# Having the right representation of the data is often crucial for supervised or unsupervised learning to succeed, 
# and preprocessing and decomposition methods play an important part in data preparation.

# Decomposition, manifold learning and clustering are essential tools to further your understanding of your data, 
# and can be the only way to make sense of your data in the absence of supervision information. 
# Even in the supervised setting, exploratory tools are important for a better understanding of the properties of the data.

# Often it is hard to quantify the usefulness of an unsupervised algorithm, 
# though this shouldn’t deter you from using them to create insights from your data.
# With these methods under your belt, 
# you are now equipped with all the essential learning algorithms that machine learning practitioners use every day.





