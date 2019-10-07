# Introduction to Machine Learning with Python
# Chapter 4: Representing Data and Engineering Features
# Refreshed by Haowen Huang

from preamble import *

import mglearn
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder

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
# plt.vlines(kb.bin_edges_[0], -3, 3, linewidth=1, alpha=.2)
# plt.legend(loc="best")
# plt.ylabel("Regression output")
# plt.xlabel("Input feature")

# The green line and red line are exactly on top of each other, 
# meaning the linear regression model and the decision tree make exactly the same predictions. 
# The linear model benefited greatly in expressiveness from the transformation of the data.

# If there are good reasons to use a linear model for a particular data set, 
# say because it is very large and high-dimensional, but some features have non-linear relations with the output, 
# binning can be a great way to increase modelling power.












