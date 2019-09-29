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
pip install numpy scipy scikit-learn matplotlib pandas pillow graphviz
$ pip install numpy scipy scikit-learn matplotlib pandas pillow graphviz
Requirement already satisfied: numpy in /usr/local/lib/python3.7/site-packages (1.17.0)
Requirement already satisfied: scipy in /usr/local/lib/python3.7/site-packages (1.3.1)
Requirement already satisfied: scikit-learn in /Users/eunice/Library/Python/3.7/lib/python/site-packages (0.21.3)
Requirement already satisfied: matplotlib in /usr/local/lib/python3.7/site-packages (3.1.1)
Requirement already satisfied: pandas in /usr/local/lib/python3.7/site-packages (0.25.0)
Requirement already satisfied: pillow in /Users/eunice/Library/Python/3.7/lib/python/site-packages (6.1.0)
Requirement already satisfied: graphviz in /Users/eunice/Library/Python/3.7/lib/python/site-packages (0.8.4)
Requirement already satisfied: joblib>=0.11 in /Users/eunice/Library/Python/3.7/lib/python/site-packages (from scikit-learn) (0.13.2)
Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.7/site-packages (from matplotlib) (1.1.0)
Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.7/site-packages (from matplotlib) (2.8.0)
Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.7/site-packages (from matplotlib) (2.4.2)
Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.7/site-packages (from matplotlib) (0.10.0)
Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.7/site-packages (from pandas) (2019.2)
Requirement already satisfied: setuptools in /usr/local/lib/python3.7/site-packages (from kiwisolver>=1.0.1->matplotlib) (41.0.1)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.7/site-packages (from python-dateutil>=2.1->matplotlib) (1.12.0)

You also need to install the graphiz C-library, which is easiest using a package manager.
If you are using OS X and homebrew, you can ``brew install graphviz``. 
$ brew install graphviz
Updating Homebrew...
==> Installing dependencies for graphviz: libpng, freetype, fontconfig, jpeg, libtiff, webp, gd, libffi, pcre, openssl@1.1, readline, python, glib, jasper, netpbm, gts and libtool
Python has been installed as
  /usr/local/bin/python3

Unversioned symlinks `python`, `python-config`, `pip` etc. pointing to
`python3`, `python3-config`, `pip3` etc., respectively, have been installed into
  /usr/local/opt/python/libexec/bin

You can install Python packages with
  pip3 install <package>
They will install into the site-package directory
  /usr/local/lib/python3.7/site-packages

For the chapter on text processing you also need to install ``nltk`` and ``spacy``:
pip install nltk spacy

### Downloading English language model
For the text processing chapter, you need to download the English language model for spacy using
python -m spacy download en

My github for ML with Python:
https://github.com/hanyun2019/Machine-Learning-with_Python.git
Local Github: /Users/research/MLPython

Create a new repository on the command line
echo "# Machine-Learning-with_Python" >> README.md
git init
git add README.md
git commit -m "first commit"
git remote add origin https://github.com/hanyun2019/Machine-Learning-with_Python.git
git push -u origin master

Push an existing repository from the command line
git remote add origin https://github.com/hanyun2019/Machine-Learning-with_Python.git
git push -u origin master

Git operation demos are as the follows:
$ echo "# Machine-Learning-with_Python" >> README.md
$ git init
Initialized empty Git repository in /Users/research/MLPython/.git/
$ git add README.md
$ git config --global user.email hxxxxxxx9@gmail.com
$ git commit -m "first commit"
$ git remote add origin https://github.com/hanyun2019/Machine-Learning-with_Python.git
$ git push -u origin master



Chapter 1: Introduction 
Installing Scikit-learn 
Scikit-learn depends on two other Python packages, NumPy and SciPy. For plotting and interactive development, you should also install matplotlib, IPython and the Jupyter notebook. 
