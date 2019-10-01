# Introduction to Machine Learning with Python
# Chapter 1

import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# # Versions Used in your environment
# import sys
# print("Python version:", sys.version)

# # import pandas as pd
# print("pandas version:", pd.__version__)

# import matplotlib
# print("matplotlib version:", matplotlib.__version__)

# # import numpy as np
# print("NumPy version:", np.__version__)

# import scipy as sp
# print("SciPy version:", sp.__version__)

# import IPython
# print("IPython version:", IPython.__version__)

# import sklearn
# print("scikit-learn version:", sklearn.__version__)



if __name__ == '__main__':

    
    x = np.array([[1,2,3],[4,5,6]])
    print("\nx:\n{}".format(x))


    # create a 2d numpy array with a diagonal of ones, and zeros everywhere else
    eye = np.eye(4)
    print("\nNumpy array:\n%s" % eye)

    # sparse matrix: 稀疏矩阵 - 矩阵中的元素大部分是0的矩阵
    # convert the numpy array to a scipy sparse matrix in CSR format
    # only the non-zero entries are stored 
    # CSR format: Compressed Sparse Row format
    sparse_matrix = sparse.csr_matrix(eye)
    print("\nScipy sparse CSR matrix:\n%s" % sparse_matrix)

    # COO representation
    data = np.ones(4)
    row_indices = np.arange(4)
    col_indices = np.arange(4)
    eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
    print("\nCOO representation:\n", eye_coo)

    # Generate a sequence of integers
    x1 = np.arange(20)
    # create a second array using sinus
    y = np.sin(x1)
    # The plot function makes a line chart of one array against another
    plt.plot(x1, y, marker="x")
    # plt.show()

    
    # create a simple dataset of people
    data = {'Name': ["John", "Anna", "Peter", "Linda"],
        'Location' : ["New York", "Paris", "Berlin", "London"],
        'Age' : [24, 13, 53, 33]
           }

    # IPython.display allows "pretty printing" of dataframes
    # in the Jupyter notebook
    # data_pandas = pd.DataFrame(data)
    # display(data_pandas)
    print("\nSimple dataset of people:\n{}".format(data))



    ## Classifying Iris Species
    # The data we will use for this example is the iris dataset, a classical dataset in machine learning an statistics.
    # It is included in scikit-learn in the dataset module. 
    # Iris plants dataset
    # --------------------

    # **Data Set Characteristics:**

    #     :Number of Instances: 150 (50 in each of three classes)
    #     :Number of Attributes: 4 numeric, predictive attributes and the class
    #     :Attribute Information:
    #         - sepal length in cm
    #         - sepal width in cm
    #         - petal length in cm
    #         - petal width in cm
    #         - class:
    #                 - Iris-Setosa
    #                 - Iris-Versicolour
    #                 - Iris-Virginica
                
    #     :Summary Statistics:

    #     ============== ==== ==== ======= ===== ====================
    #                     Min  Max   Mean    SD   Class Correlation
    #     ============== ==== ==== ======= ===== ====================
    #     sepal length:   4.3  7.9   5.84   0.83    0.7826
    #     sepal width:    2.0  4.4   3.05   0.43   -0.4194
    #     petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
    #     petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
    #     ============== ==== ==== ======= ===== ====
    #     :Missing Attribute Values: None
    #     :Class Distribution: 33.3% for each of 3 classes.
    #     The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant.  
    #     One class is linearly separable from the other 2; the latter are NOT linearly separable from each other.

    print("\n--------'Sample - Classifying Iris Species' starts here:--------")
    
    ## Meet the Data
    # from sklearn.datasets import load_iris
    iris_dataset = load_iris()
    print("\nKeys of iris_dataset:\n", format(iris_dataset.keys()))

    # The value to the key DESCR is a short description of the dataset. 
    # We show the beginning of the description here.
    # print(iris_dataset['DESCR'][:9993] + "\n...")

    ## Target names:
    # The value with key target_names is an array of strings, 
    # containing the species of flower that we want to predict.
    # Target names: ['setosa' 'versicolor' 'virginica']
    print("\nTarget names:\n", iris_dataset['target_names'])

    ## Feature names:
    # The feature_names are a list of strings, giving the description of each feature.
    print("\nFeature names:\n", iris_dataset['feature_names'])

    ## Data:
    # The data itself is contained in the target and data fields. 
    # The data contains the numeric measurements of sepal length, sepal width, petal length, 
    # and petal width in a numpy array. 
    print("\nType of iris data:\n", type(iris_dataset['data']))
    # Type of data: <class 'numpy.ndarray'>
     
    print("\nShape of iris data:\n", iris_dataset['data'].shape)
    # The rows in the data array correspond to flowers, 
    # while the columns represent the four measurements that were taken for each flower.
    # Shape of data: (150, 4)
    # 150: The data contains measurements for 150 different flowers.
    # 40:  Each flower has 4 measurements: sepal length, sepal width, petal length, and petal width 

    # Here are the feature values for the first five samples.
    print("\nFirst five rows of data:\n", iris_dataset['data'][:5])

    # The target array contains the species of each of the flowers that were measured, also as a numpy array:
    print("\nType of target:\n", type(iris_dataset['target']))

    # The target is a one-dimensional array, with one entry per flower. (一维数组 每朵花对应其中一个数据)
    print("\nShape of target:\n", iris_dataset['target'].shape)

    # The species are encoded as integers from 0 to 2:
    # The meaning of the numbers are given by the iris['target_names'] array: 
    # 0 means Setosa, 1 means Versicolor and 2 means Virginica.
    print("\nIris Target:\n", iris_dataset['target'])



    ## Measuring Success: Training and Testing Data
    # Scikit-learn contains a function that shuffles the dataset and splits it for you, 
    # the train_test_split function.
    # This function extracts 75% of the rows in the data as the training set, 
    # together with the corresponding labels for this data. 
    # The remaining 25% of the data, together with the remaining labels are declared as the test set.

    # To make sure that we will get the same output if we run the same function several times, 
    # we provide the pseudo random number generator with a fixed seed using the random_state parameter. 
    # This will make the outcome deterministic, so this line will always have the same outcome. 

    # from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        iris_dataset['data'], iris_dataset['target'], random_state=0)

    # The output of the train_test_split function are X_train, X_test, y_train and y_test, which are all numpy arrays. 
    # X_train contains 75% of the rows of the dataset, and X_test contains the remaining 25%.
    print("\nX_train shape:\n", X_train.shape)
    print("\ny_train shape:\n", y_train.shape)

    print("\nX_test shape:\n", X_test.shape)
    print("\ny_test shape:\n", y_test.shape)



    ## First things first: Look at your data
    # Before building a machine learning model, it is often a good idea to inspect the data, 
    # to see if the task is easily solvable without machine learning, 
    # or if the desired infor‐ mation might not be contained in the data.


    







    