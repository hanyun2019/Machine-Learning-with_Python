# Machine-Learning-with_Python


Local Github: /Users/research/MLPython

Local PDF & source code: 
/Users/MachineLearning/books/
 /Users/MachineLearning/books/introduction_to_ml_with_python


Start Jupyter:
$ jupyter notebook


# Introduction to Machine Learning with Python

If you get ``ImportError: No module named mglearn`` you can try to install mglearn into your python environment using the command ``pip install mglearn`` in your terminal or ``!pip install mglearn`` in Jupyter Notebook.
## Setup

To run the code, you need the packages ``numpy``, ``scipy``, ``scikit-learn``, ``matplotlib``, ``pandas`` and ``pillow``.
Some of the visualizations of decision trees and neural networks structures also require ``graphviz``. The chapter
on text processing also requirs ``nltk`` and ``spacy``.

### Installing packages with pip
If you already have a Python environment and are using pip to install packages, you need to run

$ pip install numpy scipy scikit-learn matplotlib pandas pillow graphviz


You also need to install the graphiz C-library, which is easiest using a package manager.
If you are using OS X and homebrew, you can ``brew install graphviz``. 

$ brew install graphviz


For the chapter on text processing you also need to install ``nltk`` and ``spacy``:

$ pip install nltk spacy

### Downloading English language model

For the text processing chapter, you need to download the English language model for spacy using

$ python -m spacy download en

My github for ML with Python:
https://github.com/hanyun2019/Machine-Learning-with_Python.git

Local Github: /Users/research/MLPython

Create a new repository on the command line

$ echo "# Machine-Learning-with_Python" >> README.md

$ git init

$ git add README.md

$ git config --global user.email xxxxxxxxx@gmail.com

$ git commit -m "first commit"

$ git remote add origin https://github.com/hanyun2019/Machine-Learning-with_Python.git

$ git push -u origin master


Push an existing repository from the command line

$ git remote add origin https://github.com/hanyun2019/Machine-Learning-with_Python.git

$ git add .

$ git commit -m "update for git commit test"

$ git push origin master


-----------------------------------------------------------


Chapter 1: Introduction 

Installing Scikit-learn 
Scikit-learn depends on two other Python packages, NumPy and SciPy. For plotting and interactive development, you should also install matplotlib, IPython and the Jupyter notebook. 


BUG FIX No 1.1
https://stackoverflow.com/questions/42592493/displaying-pair-plot-in-pandas-data-frame

grr = pd.plotting.scatter_matrix(iris_dataframe, c=Y, figsize=(15, 15), marker='o',
                                 hist_kwds={'bins': 20}, s=60, alpha=.8)

Thanks to michael-szczepaniak for pointing out that this API had been deprecated.

grr = pd.scatter_matrix(iris_dataframe, c=Y, figsize=(15, 15), marker='o',
                        hist_kwds={'bins': 20}, s=60, alpha=.8)

I just had to remove the cmap=mglearn.cm3 piece, because I was not able to make mglearn work. There is a version mismatch issue with sklearn.

To not display the image and save it directly to file you can use this method:

plt.savefig('foo.png')

Also remove

%matplotlib inline

-----------------------------------------------------------


Chapter 2: Supervised Learning

------------------------------------

BUG FIX 2.1

/Users/eunice/Library/Python/3.7/lib/python/site-packages/sklearn/externals/joblib/__init__.py:16: DeprecationWarning: sklearn.externals.joblib is deprecated in 0.21 and will be removed in 0.23. Please import this functionality directly from joblib, which can be installed with: pip install joblib. If this warning is raised when loading pickled models, you may need to re-serialize those models with scikit-learn 0.21+.

--------------------

for File "/Users/eunice/Library/Python/3.7/lib/python/site-packages/sklearn/externals/joblib/__init__.py":

from joblib import *

Change to: 

import joblib

--------------------

for File "mglearn/plot_nmf.py":

from sklearn.externals.joblib import Memory

Change to:

from joblib import Memory

---------------------

for File "mglearn/plot_pca.py":

from sklearn.externals.joblib import Memory

Change to:

from joblib import Memory

------------------------------------

BUG FIX 2.2

/Users/eunice/Library/Python/3.7/lib/python/site-packages/sklearn/externals/six.py:31: DeprecationWarning: The module is deprecated in version 0.21 and will be removed in version 0.23 since we've dropped support for Python 2.7. Please rely on the official version of six (https://pypi.org/project/six/). 
"(https://pypi.org/project/six/).", DeprecationWarning)

--------------------

for File "/Users/eunice/Library/Python/3.7/lib/python/site-packages/sklearn/externals/six.py":

Comment the follow code:

import warnings

warnings.warn("The module is deprecated in version 0.21 and will be removed "

               "in version 0.23 since we've dropped support for Python 2.7. "

               "Please rely on the official version of six "

               "(https://pypi.org/project/six/).", DeprecationWarning)

------------------------------------

BUG FIX 2.3

/Users/eunice/Library/Python/3.7/lib/python/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)

https://stackoverflow.com/questions/52670012/convergencewarning-liblinear-failed-to-converge-increase-the-number-of-iterati

LinearSVC default max_iter is 1000, change it to a large value.

--------------------

For my writing code: ch02/Supervised.py:

LinearSVC()

Change to:

LinearSVC(max_iter=10000)


------------------------------------

BUG FIX 2.4

/Users/eunice/Library/Python/3.7/lib/python/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)

--------------------

For my writing code: ch02/Supervised.py:

LogisticRegression()

Change to:

LogisticRegression(solver='lbfgs')


------------------------------------

BUG FIX 2.5

/Users/eunice/Library/Python/3.7/lib/python/site-packages/sklearn/linear_model/logistic.py:947: ConvergenceWarning: lbfgs failed to converge. Increase the number of iterations.
  "of iterations.", ConvergenceWarning)

https://github.com/scikit-learn/scikit-learn/pull/11476

--------------------

For my writing code: ch02/Supervised.py:

LogisticRegression(solver='lbfgs')

Change to:

LogisticRegression(solver='lbfgs',max_iter=5000)

------------------------------------

BUG FIX 2.6

As solver='lbfgs' doesn't supoort "l1" penalty, 

So when use: 

LogisticRegression(C=C, penalty="l1")

I change to: 

LogisticRegression(C=C, solver="liblinear",penalty="l1",max_iter=5000)

Because solver="liblinear" can support "l1" penalty.

------------------------------------

BUG FIX 2.7

/Users/eunice/Library/Python/3.7/lib/python/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)

from sklearn.svm import SVC 

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

svc = SVC()

svc.fit(X_train, y_train)

......

---------

Change "svc = SVC()" to:

svc = SVC(gamma='scale')

------------------------------------

BUG FIX 3.1

/Users/eunice/Library/Python/3.7/lib/python/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.

So as: 

svm = SVC()

Change to:

svm = SVC(gamma='auto')

------------------------------------














