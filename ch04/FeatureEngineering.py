# Introduction to Machine Learning with Python
# Chapter 4: Representing Data and Engineering Features
# Refreshed by Haowen Huang

from preamble import *

import mglearn
import os

from sklearn.datasets import load_boston
from sklearn.datasets import load_breast_cancer

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

# The question of how to represent your data best for a particular application is known as feature engineering, 
# and it is one of the main tasks of data scientists and machine learning practitioners trying to solve real-world problems. 
# Representing your data in the right way can have a bigger influence on the performance of a supervised model than the exact parameters you choose.

## 4.1 Categorical Variables    分类变量
# As an example, we will use the dataset of adult incomes in the United States, derived from the 1994 census database.
# The task of the adult dataset is to predict whether a worker has an income of over 50.000$ or under 50.000$.

# In this dataset, age and hours-per-week are continuous features, which we know how to treat. 
# The workclass, education, sex and ocupation features are categorical, 
# however. All of them come from a fixed list of possible values, as opposed to a range, and denote a qualitative property, as opposed to a quantity.


## 4.1.1 One-Hot-Encoding (Dummy variables)     One-Hot编码(虚拟变量)

print("\n----------- One-Hot-Encoding (Dummy variables) - U.S. adult incomes dataset example -----------")

# import os
# The file has no headers naming the columns, so we pass header=None
# and provide the column names explicitly in "names"
adult_path = os.path.join(mglearn.datasets.DATA_PATH, "adult.data")
data = pd.read_csv(
    adult_path, header=None, index_col=False,
    names=['age', 'workclass', 'fnlwgt', 'education',  'education-num',
           'marital-status', 'occupation', 'relationship', 'race', 'gender',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
           'income'])
# For illustration purposes, we only select some of the columns
data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week',
             'occupation', 'income']]
# IPython.display allows nice output formatting within the Jupyter notebook
display(data.head())

# Result:
#  age          workclass   education   gender  hours-per-week  \
# 0   39          State-gov   Bachelors     Male              40   
# 1   50   Self-emp-not-inc   Bachelors     Male              13   
# 2   38            Private     HS-grad     Male              40   
# 3   53            Private        11th     Male              40   
# 4   28            Private   Bachelors   Female              40   

#            occupation  income  
# 0        Adm-clerical   <=50K  
# 1     Exec-managerial   <=50K  
# 2   Handlers-cleaners   <=50K  
# 3   Handlers-cleaners   <=50K  
# 4      Prof-specialty   <=50K  


# Checking string-encoded categorical data
print(data.gender.value_counts())
# Result:
# Male      21790
# Female    10771
# Name: gender, dtype: int64

# There are two ways to convert your data to a one-hot encoding of categorical variables, either using pandas or using scikit-learn. 
# At the time of writing, using pan das for this setting is slightly easier
print("\nOriginal features:\n", list(data.columns), "\n")
data_dummies = pd.get_dummies(data)
print("\nFeatures after get_dummies:\n", list(data_dummies.columns))
# Result:
# Original features:
#  ['age', 'workclass', 'education', 'gender', 'hours-per-week', 'occupation', 'income'] 

# Features after get_dummies:
#  ['age', 'hours-per-week', 'workclass_ ?', 'workclass_ Federal-gov', 'workclass_ Local-gov', 'workclass_ Never-worked', 'workclass_ Private', 'workclass_ Self-emp-inc', 'workclass_ Self-emp-not-inc', 'workclass_ State-gov', 'workclass_ Without-pay', 'education_ 10th', 'education_ 11th', 'education_ 12th', 'education_ 1st-4th', 'education_ 5th-6th', 'education_ 7th-8th', 'education_ 9th', 'education_ Assoc-acdm', 'education_ Assoc-voc', 'education_ Bachelors', 'education_ Doctorate', 'education_ HS-grad', 'education_ Masters', 'education_ Preschool', 'education_ Prof-school', 'education_ Some-college', 'gender_ Female', 'gender_ Male', 'occupation_ ?', 'occupation_ Adm-clerical', 'occupation_ Armed-Forces', 'occupation_ Craft-repair', 'occupation_ Exec-managerial', 'occupation_ Farming-fishing', 'occupation_ Handlers-cleaners', 'occupation_ Machine-op-inspct', 'occupation_ Other-service', 'occupation_ Priv-house-serv', 'occupation_ Prof-specialty', 'occupation_ Protective-serv', 'occupation_ Sales', 'occupation_ Tech-support', 'occupation_ Transport-moving', 'income_ <=50K', 'income_ >50K']

display(data_dummies.head())
# Result:
#  age  hours-per-week  workclass_ ?  workclass_ Federal-gov  ...  \
# 0   39              40             0                       0  ...   
# 1   50              13             0                       0  ...   
# 2   38              40             0                       0  ...   
# 3   53              40             0                       0  ...   
# 4   28              40             0                       0  ...   

#    occupation_ Tech-support  occupation_ Transport-moving  income_ <=50K  \
# 0                         0                             0              1   
# 1                         0                             0              1   
# 2                         0                             0              1   
# 3                         0                             0              1   
# 4                         0                             0              1   

#    income_ >50K  
# 0             0  
# 1             0  
# 2             0  
# 3             0  
# 4             0  

# 下面我们使用 value 属性将 data_dummies 数据框(DataFrame)转换为 Numpy 数组，然后在其上训练一个机器学习模型
# 在训练模型之前，注意把目标变量(现在被编码为两个 income 列)从数据中分离出来
# 将输出变量或其中的一些导出属性包含在特征表示中，这是构建监督机器学习模型时一个非常常见的错误
print("\ndata_dummies:\n", data_dummies)
features = data_dummies.loc[:, 'age':'occupation_ Transport-moving']
print("\nfeatures:\n", features)

# Extract NumPy arrays
X = features.values
y = data_dummies['income_ >50K'].values
print("\nX.shape: {}  y.shape: {}".format(X.shape, y.shape))
# Result:
# X.shape: (32561, 44)  y.shape: (32561,)

# 现在数据的表示方式可以被 scikit_learn 处理
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
logreg = LogisticRegression(solver='lbfgs',max_iter=5000)
logreg.fit(X_train, y_train)
print("\nTest score: {:.2f}".format(logreg.score(X_test, y_test)))
# Result:
# Test score: 0.81


## 4.1.2 Numbers Can Encode Categoricals    数字可以编码分类变量
# 在前面的 adult 数据集例子中，分类变量被编码为字符串
# 无论是为了便于存储还是因为数据的收集方式，分类变量通常被编码为整数(例如人口普查问卷中在某些框打勾等)
# 
# pandas 的 get_dummies 函数将所有数字看作是连续的，不会为其创建虚拟变量
# 为了解决这个问题，可以使用 scikit-learn 的 OneHotEncoder，指定变量是连续还是离散的，也可以将数据框中的数值列转换为字符串 
print("\n----------- Numbers Can Encode Categoricals(pandas.get_dummies()) - example -----------")

# create a DataFrame with an integer feature and a categorical string feature
demo_df = pd.DataFrame({'Integer Feature': [0, 1, 2, 1],
                        'Categorical Feature': ['socks', 'fox', 'socks', 'box']})
display(demo_df)
# Result:
#           Integer Feature   Categorical Feature
# 0                0               socks
# 1                1                 fox
# 2                2               socks
# 3                1                 box

# 使用 get_dummies 只会编码字符串特征，不会改变整数特征
display(pd.get_dummies(demo_df))
# Result:
#           Integer Feature  Categorical Feature_box  Categorical Feature_fox  \
# 0                0                        0                        0   
# 1                1                        0                        1   
# 2                2                        0                        0   
# 3                1                        1                        0   

#                Categorical Feature_socks  
# 0                          1  
# 1                          0  
# 2                          1  
# 3                          0  

demo_df['Integer Feature'] = demo_df['Integer Feature'].astype(str)
display(pd.get_dummies(demo_df, columns=['Integer Feature', 'Categorical Feature']))
# Result:
#           Integer Feature_0  Integer Feature_1  Integer Feature_2  \
# 0                  1                  0                  0   
# 1                  0                  1                  0   
# 2                  0                  0                  1   
# 3                  0                  1                  0   

#               Categorical Feature_box  Categorical Feature_fox  Categorical Feature_socks  
# 0                        0                        0                          1  
# 1                        0                        1                          0  
# 2                        0                        0                          1  
# 3                        1                        0                          0  



# OneHotEncoder and ColumnTransformer: Categorical Variables with scikit-learn

print("\n----------- Numbers Can Encode Categoricals(scikit-learn: OneHotEncoder and ColumnTransformer) - example -----------")

# from sklearn.preprocessing import OneHotEncoder
# Setting sparse=False means OneHotEncode will return a numpy array, not a sparse matrix
ohe = OneHotEncoder(sparse=False)
print(ohe.fit_transform(demo_df))
# Result:
# [[1. 0. 0. 0. 0. 1.]
#  [0. 1. 0. 0. 1. 0.]
#  [0. 0. 1. 0. 0. 1.]
#  [0. 1. 0. 1. 0. 0.]]

print(ohe.get_feature_names())
# Result:
# ['x0_0' 'x0_1' 'x0_2' 'x1_box' 'x1_fox' 'x1_socks']

display(data.head())
# Result:
#    age          workclass   education   gender  hours-per-week  \
# 0   39          State-gov   Bachelors     Male              40   
# 1   50   Self-emp-not-inc   Bachelors     Male              13   
# 2   38            Private     HS-grad     Male              40   
# 3   53            Private        11th     Male              40   
# 4   28            Private   Bachelors   Female              40   

#            occupation  income  
# 0        Adm-clerical   <=50K  
# 1     Exec-managerial   <=50K  
# 2   Handlers-cleaners   <=50K  
# 3   Handlers-cleaners   <=50K  
# 4      Prof-specialty   <=50K 

# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler
ct = ColumnTransformer(
    [("scaling", StandardScaler(), ['age', 'hours-per-week']),
     ("onehot", OneHotEncoder(sparse=False), ['workclass', 'education', 'gender', 'occupation'])])

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# get all columns apart from income for the features
data_features = data.drop("income", axis=1)
# split dataframe and income
X_train, X_test, y_train, y_test = train_test_split(
    data_features, data.income, random_state=0)

ct.fit(X_train)
X_train_trans = ct.transform(X_train)
print(X_train_trans.shape)
# Result: 
# (24420, 44)

logreg = LogisticRegression(solver='lbfgs',max_iter=5000)
logreg.fit(X_train_trans, y_train)

X_test_trans = ct.transform(X_test)
print("\nTest score: {:.2f}".format(logreg.score(X_test_trans, y_test)))
# Result:
# Test score: 0.81

print("\nct.named_transformers_.onehot:\n",ct.named_transformers_.onehot)
# Result:
# ct.named_transformers_.onehot:
#  OneHotEncoder(categorical_features=None, categories=None, drop=None,
#               dtype=<class 'numpy.float64'>, handle_unknown='error',
#               n_values=None, sparse=False)


## 4.2 Binning, Discretization, Linear Models and Trees     分箱、离散化、线性模型和树

print("\n----------- Binning, Discretization, Linear Models and Trees - wave dataset example -----------")

# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor

X, y = mglearn.datasets.make_wave(n_samples=120)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)

reg = DecisionTreeRegressor(min_samples_leaf=3).fit(X, y)
# plt.plot(line, reg.predict(line), label="decision tree")

reg = LinearRegression().fit(X, y)
# plt.plot(line, reg.predict(line), label="linear regression")

# plt.plot(X[:, 0], y, 'o', c='k')
# plt.ylabel("Regression output")
# plt.xlabel("Input feature")
# plt.legend(loc="best")


# As you know, linear models can only model linear relationships, which are lines in the case of a single feature. 
# The decision tree an build a much more complex model of the data. However, this is strongly dependent on our representation of the data. 
# One way to make linear models more powerful on continuous data is to using binning (also known as discretization) of the feature 
# to split it up into multiple features as follows:

# We imagine a partition of the input range of -3 to 3 of this feature into a fixed number of bins. 
# Here, we pass bin boundaries from -3 to 3 with 11 equally sized steps. 
# An array of 11 bin boundaries will create 10 bins - they are the space in between two consecutive boundaries.

bins = np.linspace(-3, 3, 11)
print("bins: {}".format(bins))
# Result:
# bins: [-3.  -2.4 -1.8 -1.2 -0.6  0.   0.6  1.2  1.8  2.4  3. ]

which_bin = np.digitize(X, bins=bins)
print("\nData points:\n", X[:5])
print("\nBin membership for data points:\n", which_bin[:5])
# Result:
# Data points:
#  [[-0.753]
#  [ 2.704]
#  [ 1.392]
#  [ 0.592]
#  [-2.064]]

# Bin membership for data points:
#  [[ 4]
#  [10]
#  [ 8]
#  [ 6]
#  [ 2]]

# from sklearn.preprocessing import OneHotEncoder

# transform using the OneHotEncoder.
encoder = OneHotEncoder(sparse=False,categories='auto')
# encoder.fit finds the unique values that appear in which_bin
encoder.fit(which_bin)
# transform creates the one-hot encoding
X_binned = encoder.transform(which_bin)
print("\nX_binned[:5]:\n",X_binned[:5])
# Result:
# X_binned[:5]:
#  [[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
#  [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]]

print("\nX_binned.shape: {}".format(X_binned.shape))
# Result:
# X_binned.shape: (120, 10)


line_binned = encoder.transform(np.digitize(line, bins=bins))
reg = LinearRegression().fit(X_binned, y)
# plt.plot(line, reg.predict(line_binned), label='linear regression binned')

reg = DecisionTreeRegressor(min_samples_split=3).fit(X_binned, y)
# plt.plot(line, reg.predict(line_binned), label='decision tree binned')
# plt.plot(X[:, 0], y, 'o', c='k')
# plt.vlines(bins, -3, 3, linewidth=1, alpha=.2)
# plt.legend(loc="best")
# plt.ylabel("Regression output")
# plt.xlabel("Input feature")

# The green line and red line are exactly on top of each other, 
# meaning the linear regression model and the decision tree make exactly the same predictions. 
# The linear model benefited greatly in expressiveness from the transformation of the data.

# If there are good reasons to use a linear model for a particular data set, 
# say because it is very large and high-dimensional, but some features have non-linear relations with the output, 
# binning can be a great way to increase modelling power.



## 4.3 Interactions and Polynomials     交互特征和多项式特征
# Another way to enrich a feature representation, in particular for linear models, 
# is adding interaction features and polynomial features of the original data. 
# This kind of feature engineering is often used in statistical modelling, 
# but also common in many practical machine learning applications.
print("\n----------- Interactions and Polynomials -----------")
print("\n----------- Interactions -----------")

# linear models can not only learn offsets, but also slopes. 
# One way to add a slope to the linear model on the binned data, is to add the original feature.
X_combined = np.hstack([X, X_binned])
print("\nX_combined.shape:\n",X_combined.shape)
print("\nX_combined[:5]:\n",X_combined[:5])
# Result:
# X_combined.shape:
#  (120, 11)

reg = LinearRegression().fit(X_combined, y)
# line_combined = np.hstack([line, line_binned])
# plt.plot(line, reg.predict(line_combined), label='linear regression combined')
# plt.vlines(bins, -3, 3, linewidth=1, alpha=.2)
# plt.legend(loc="best")
# plt.ylabel("Regression output")
# plt.xlabel("Input feature")
# plt.plot(X[:, 0], y, 'o', c='k')

# Now, the model learned an offset for each bin, together with a slope. 
# There is the single x-axis feature which has a single slope. 
# We would rather have a separate slope for each bin! 
# We can achieve this by adding an interaction or product feature that indicates in which bin a data‐point is and where it lies on the x-axis.

# This feature is a product of the bin-indicator and the original feature. Let’s create this dataset:
X_product = np.hstack([X_binned, X * X_binned])
print(X_product.shape)
print("\nX_product.shape:\n",X_product.shape)
print("\nX_product[:5]:\n",X_product[:5])
# Result:
# X_product.shape:
# (120, 20)

# This dataset not has 20 features: the indicator for which bin a data point is in, and a product of the original feature and the bin indicator. 
# You can think of the product feature as a separate copy of the x-axis feature for each bin. 
# It is the original feature within the bin, and zero everywhere else.
# 你可以将乘积特征看作每个箱子x轴特征的单独副本，它在箱子内等于原始特征，在其他位置等于零

reg = LinearRegression().fit(X_product, y)

line_product = np.hstack([line_binned, line * line_binned])
# plt.plot(line, reg.predict(line_product), label='linear regression product')
# plt.vlines(bins, -3, 3, linewidth=1, alpha=.2)
# plt.plot(X[:, 0], y, 'o', c='k')
# plt.ylabel("Regression output")
# plt.xlabel("Input feature")
# plt.legend(loc="best")

# As you can see, now each bin has its own offset and slope in this model.

# Using binning is one way to expand a continuous feature. 
# Another one is to use polynomials of the original features.
print("\n----------- Polynomials -----------")

# For a given featurex, we might want to considerx ** 2,x ** 3,x ** 4and so on. 
# This is implemented in PolynomialFeatures in the preprocessing module。

# from sklearn.preprocessing import PolynomialFeatures

# include polynomials up to x ** 10:
# the default "include_bias=True" adds a feature that's constantly 1
poly = PolynomialFeatures(degree=10, include_bias=False)
poly.fit(X)
X_poly = poly.transform(X)
print("\nX_poly.shape: {}".format(X_poly.shape))
print("\nX_poly[:5]: {}".format(X_poly[:5]))
# Result:
# X_poly.shape: (120, 10)

# 我们比较 X_poly 和 X 的元素
print("Entries of X:\n{}".format(X[:5]))
print("Entries of X_poly:\n{}".format(X_poly[:5]))

# 你可以通过调用 get_feature_names 方法来获取特征的语义，给出每个特征的指数
print("\nPolynomial feature names:\n{}".format(poly.get_feature_names()))
# Result:
# Polynomial feature names:
# ['x0', 'x0^2', 'x0^3', 'x0^4', 'x0^5', 'x0^6', 'x0^7', 'x0^8', 'x0^9', 'x0^10']


# Using polynomial features together with a linear regression model yields the classical model of polynomial regression.
# 将多项式特征与线性回归模型一起使用，可以得到经典的多项式回归(polynomial regression)模型
print("\n----------- Polynomials Regression -----------")

reg = LinearRegression().fit(X_poly, y)
line_poly = poly.transform(line)
# plt.plot(line, reg.predict(line_poly), label='polynomial linear regression')
# plt.plot(X[:, 0], y, 'o', c='k')
# plt.ylabel("Regression output")
# plt.xlabel("Input feature")
# plt.legend(loc="best")

# As you can see, polynomial feature yield a very smooth fit on this one-dimensional data. 
# However, polynomials of high degree tend to behave in extreme ways on the boundaries or in regions of little data.


# As a comparison, here is a kernel SVM model learned on the original data, without any transformation:

# from sklearn.svm import SVR

for gamma in [1, 10]:
    svr = SVR(gamma=gamma).fit(X, y)
#    plt.plot(line, svr.predict(line), label='SVR gamma={}'.format(gamma))

# plt.plot(X[:, 0], y, 'o', c='k')
# plt.ylabel("Regression output")
# plt.xlabel("Input feature")
# plt.legend(loc="best")

# Using a more complex model, a kernel SVM, 
# we are able to learn a similarly complex prediction to the polynomial regression without using any transformations of the features.


# As a more realistic application of interactions and polynomials, let’s look again at the Boston Housing data set. 
# We already used polynomial features on this dataset in Chapter 2. 
# Now let us have a look at how these features were constructed, and at how much the polynomial features help. 
# First we load the data, and rescale it to be between 0 and 1 using MinMaxScaler.
print("\n----------- Interactions and Polynomials: the Boston Housing dataset -----------")

# from sklearn.datasets import load_boston
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler

boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=0)

# rescale data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

poly = PolynomialFeatures(degree=2).fit(X_train_scaled)
X_train_poly = poly.transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
print("\nX_train.shape: {}".format(X_train.shape))
print("\nX_train_poly.shape: {}".format(X_train_poly.shape))
# Result:
# X_train.shape: (379, 13)
# X_train_poly.shape: (379, 105)

# The data originally had 13 features, which were expanded into 105 interaction features. 
# These new features represent all possible interactions between two different original features, as well as the square of each original feature. 
# degree=2 here means that we look at all features that are the product of up to two original features. 
# The exact correspondence between input and output features can be found using the get_feature_names method:

print("\nPolynomial feature names:\n{}".format(poly.get_feature_names()))
# Result:
# Polynomial feature names:
# ['1', 'x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x0^2', 'x0 x1', 'x0 x2', 'x0 x3', 'x0 x4', 'x0 x5', 'x0 x6', 'x0 x7', 'x0 x8', 'x0 x9', 'x0 x10', 'x0 x11', 'x0 x12', 'x1^2', 'x1 x2', 'x1 x3', 'x1 x4', 'x1 x5', 'x1 x6', 'x1 x7', 'x1 x8', 'x1 x9', 'x1 x10', 'x1 x11', 'x1 x12', 'x2^2', 'x2 x3', 'x2 x4', 'x2 x5', 'x2 x6', 'x2 x7', 'x2 x8', 'x2 x9', 'x2 x10', 'x2 x11', 'x2 x12', 'x3^2', 'x3 x4', 'x3 x5', 'x3 x6', 'x3 x7', 'x3 x8', 'x3 x9', 'x3 x10', 'x3 x11', 'x3 x12', 'x4^2', 'x4 x5', 'x4 x6', 'x4 x7', 'x4 x8', 'x4 x9', 'x4 x10', 'x4 x11', 'x4 x12', 'x5^2', 'x5 x6', 'x5 x7', 'x5 x8', 'x5 x9', 'x5 x10', 'x5 x11', 'x5 x12', 'x6^2', 'x6 x7', 'x6 x8', 'x6 x9', 'x6 x10', 'x6 x11', 'x6 x12', 'x7^2', 'x7 x8', 'x7 x9', 'x7 x10', 'x7 x11', 'x7 x12', 'x8^2', 'x8 x9', 'x8 x10', 'x8 x11', 'x8 x12', 'x9^2', 'x9 x10', 'x9 x11', 'x9 x12', 'x10^2', 'x10 x11', 'x10 x12', 'x11^2', 'x11 x12', 'x12^2']


# Let’s compare the performance using Ridge on the data with and without interactions:

# from sklearn.linear_model import Ridge

ridge = Ridge().fit(X_train_scaled, y_train)
print("\nScore without interactions(Ridge): {:.3f}".format(ridge.score(X_test_scaled, y_test)))
ridge = Ridge().fit(X_train_poly, y_train)
print("\nScore with interactions(Ridge): {:.3f}".format(ridge.score(X_test_poly, y_test)))
# Result:
# Score without interactions: 0.621
# Score with interactions: 0.753

# Clearly the interactions and polynomial features gave us a good boost in performance when using Ridge. 
# When using a more complex model like a random forest, the story is a bit different.

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100).fit(X_train_scaled, y_train)
print("\nScore without interactions(RandomForestRegressor): {:.3f}".format(rf.score(X_test_scaled, y_test)))
rf = RandomForestRegressor(n_estimators=100).fit(X_train_poly, y_train)
print("\nScore with interactions(RandomForestRegressor): {:.3f}".format(rf.score(X_test_poly, y_test)))
# Result:
# Score without interactions(RandomForestRegressor): 0.794
# Score with interactions(RandomForestRegressor): 0.776

# You can see that even without additional features, the random forest beats the performance of Ridge. 
# Adding interactions and polynomials actually decreases performance slightly.



## 4.4 Univariate Nonlinear Transformations     单变量非线性变换
#
print("\n----------- Univariate Nonlinear Transformations -----------")

rnd = np.random.RandomState(0)
X_org = rnd.normal(size=(1000, 3))
w = rnd.normal(size=3)

X = rnd.poisson(10 * np.exp(X_org))
y = np.dot(X_org, w)

print("\nNumber of feature appearances:\n{}".format(np.bincount(X[:, 0])))
bins = np.bincount(X[:, 0])
# plt.bar(range(len(bins)), bins, color='grey')
# plt.ylabel("Number of appearances")
# plt.xlabel("Value")

# from sklearn.linear_model import Ridge

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
score = Ridge().fit(X_train, y_train).score(X_test, y_test)
print("\nTest score(Ridge): {:.3f}".format(score))
# Result:
# Test score: 0.622

# 你可以从相对较小的R^2分数中看出：Ridge无法真正捕捉到x和y之间的关系
# 应用对数变换可能有用

X_train_log = np.log(X_train + 1)
X_test_log = np.log(X_test + 1)
# plt.hist(X_train_log[:, 0], bins=25, color='gray')
# plt.ylabel("Number of appearances")
# plt.xlabel("Value")
score = Ridge().fit(X_train_log, y_train).score(X_test_log, y_test)
print("\nTest score(Ridge with logarithmic transformation): {:.3f}".format(score))
# Result:
# Test score(Ridge with logarithmic transformation): 0.875

# 对数变换之后，数据分布的不对称性变小，也不再有非常大的异常值


# 为数据集和模型的所有组合寻找最佳变换是一种艺术。
# Finding the transformation that works best for each combination of dataset and model is somewhat of an art. 

# Summary:
# a) Binning, polynomials and interactions can have a huge influence on how models perform on a given dataset. 
#    This is in particularly true for less complex models like linear models and naive Bayes.
# b) Tree-based models on the other hand are often able to discover important interactions themselves, 
#    and don’t require transforming the data explicitly most of the time.
# c) Other models like SVMs, nearest neighbors and neural networks might sometimes benefit from using binning, interactions or polynomials, 
#    but the implications there are usually much less clear than in the case of linear models. 


## 4.5 Automatic Feature Selection
# Automatic Feature Selection:
# a) Univariate statistics  单变量统计
# b) model-based selection  基于模型的选择
# c) iterative selection    迭代选择

## 4.5.1 Univariate statistics  单变量统计
print("\n----------- Automatic Feature Selection: Univariate statistics on breast_cancer dataset -----------")

# from sklearn.datasets import load_breast_cancer
# from sklearn.feature_selection import SelectPercentile
# from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

# get deterministic random numbers
rng = np.random.RandomState(42)
noise = rng.normal(size=(len(cancer.data), 50))
# add noise features to the data
# the first 30 features are from the dataset, the next 50 are noise
X_w_noise = np.hstack([cancer.data, noise])

X_train, X_test, y_train, y_test = train_test_split(X_w_noise, cancer.target, random_state=0, test_size=.5)

# use f_classif (the default) and SelectPercentile to select 50% of features
select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)
# transform training set
X_train_selected = select.transform(X_train)

print("\nX_train.shape: {}".format(X_train.shape))
print("\nX_train_selected.shape: {}".format(X_train_selected.shape))
# Result:
# X_train.shape: (284, 80)
# X_train_selected.shape: (284, 40)

# As you can see, the number of features was reduced from 80 to 40 (50 percent of the original number of features). 
# We can find out which features have been selected using the get_support method, which returns a boolean mask of the selected features:
mask = select.get_support()
print("\nmask:\n",mask)
# Result:
# mask:
#  [ True  True  True  True  True  True  True  True  True False  True False
#   True  True  True  True  True  True False False  True  True  True  True
#   True  True  True  True  True  True False False False  True False  True
#  False False  True False False False False  True False False  True False
#  False  True False  True False False False False False False  True False
#   True False False False False  True False  True False False False False
#   True  True False  True False False False False]

# visualize the mask. black is True, white is False
# plt.matshow(mask.reshape(1, -1), cmap='gray_r')
# plt.xlabel("Sample index")
# plt.yticks(())

# As you can see from the visualization of the mask above, most of the selected features are the original features, 
# and most of the noise features were removed. However, the recovery of the original features is not perfect.

# Let’s compare the performance of logistic regression on all features against the performance using only the selected features:

# from sklearn.linear_model import LogisticRegression

# transform test data
X_test_selected = select.transform(X_test)

# lr = LogisticRegression(solver='lbfgs',max_iter=5000)
lr = LogisticRegression()
lr.fit(X_train, y_train)
print("\nScore with all features: {:.3f}".format(lr.score(X_test, y_test)))
lr.fit(X_train_selected, y_train)
print("\nScore with only selected features: {:.3f}".format(lr.score(X_test_selected, y_test)))
# Result:
# Score with all features: 0.930
# Score with only selected features: 0.940

# In this case, removing the noise features improved performance, even though some of the original features where lost. 


## 4.5.2 model-based selection 
print("\n----------- Automatic Feature Selection: model-based selection -----------")

# from sklearn.feature_selection import SelectFromModel
# from sklearn.ensemble import RandomForestClassifier

select = SelectFromModel(
    RandomForestClassifier(n_estimators=100, random_state=42),
    threshold="median")

select.fit(X_train, y_train)
X_train_l1 = select.transform(X_train)
print("\nX_train.shape: {}".format(X_train.shape))
print("\nX_train_l1.shape: {}".format(X_train_l1.shape))
# Result: 

mask = select.get_support()
# visualize the mask. black is True, white is False
# plt.matshow(mask.reshape(1, -1), cmap='gray_r')
# plt.xlabel("Sample index")
# plt.yticks(())

X_test_l1 = select.transform(X_test)
score = LogisticRegression().fit(X_train_l1, y_train).score(X_test_l1, y_test)
print("\nTest score: {:.3f}".format(score))
# Result:
# Test score: 0.951


## 4.5.3 iterative selection
# In univariate testing, we build used no model, while in model based selection we used a single model to select features. 
# In iterative feature selection, a series of models is built, with varying numbers of features. 

# One particular method of this kind is recursive feature elimination (RFE) which starts with all features, builds a model, 
# and discards the least important feature according to the model. 
# Then, a new model is built, using all but the discarded feature, and so on, until only a pre-specified number of features is left. 

print("\n----------- Automatic Feature Selection: iterative selection -----------")

# from sklearn.feature_selection import RFE

select = RFE(RandomForestClassifier(n_estimators=100, random_state=42),n_features_to_select=40)

select.fit(X_train, y_train)
# visualize the selected features:
mask = select.get_support()
# plt.matshow(mask.reshape(1, -1), cmap='gray_r')
# plt.xlabel("Sample index")
# plt.yticks(())

# The feature selection got better compared to the univariate and model based selection, but one feature was still missed. 
# Running the above code takes significantly longer than the model based selection, because a random forest model is trained 40 times, 
# once for each feature that is dropped.
X_train_rfe = select.transform(X_train)
X_test_rfe = select.transform(X_test)

score = LogisticRegression().fit(X_train_rfe, y_train).score(X_test_rfe, y_test)
print("\nTest score(the random forest model): {:.3f}".format(score))
# Result:
# Test score(the random forest model): 0.951

# We can also use the model used inside the RFE to make predictions. This uses only the feature set that was selected:
print("\nTest score(the model used inside the RFE): {:.3f}".format(select.score(X_test, y_test)))
# Result:
# Test score(the model used inside the RFE): 0.951













