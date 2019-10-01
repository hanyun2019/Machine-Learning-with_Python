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

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn import datasets

iris_dataset = datasets.load_iris()
X = iris_dataset.data
Y = iris_dataset.target

iris_dataframe = pd.DataFrame(X, columns=iris_dataset.feature_names)

grr = pd.plotting.scatter_matrix(iris_dataframe, c=Y, figsize=(15, 15), marker='o',
                                 hist_kwds={'bins': 20}, s=60, alpha=.8)
For pandas version < v0.20.0.

Thanks to michael-szczepaniak for pointing out that this API had been deprecated.

grr = pd.scatter_matrix(iris_dataframe, c=Y, figsize=(15, 15), marker='o',
                        hist_kwds={'bins': 20}, s=60, alpha=.8)
I just had to remove the cmap=mglearn.cm3 piece, because I was not able to make mglearn work. There is a version mismatch issue with sklearn.

To not display the image and save it directly to file you can use this method:

plt.savefig('foo.png')
Also remove

%matplotlib inline

