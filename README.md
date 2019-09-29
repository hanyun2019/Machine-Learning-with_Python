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


You also need to install the graphiz C-library, which is easiest using a package manager.
If you are using OS X and homebrew, you can ``brew install graphviz``. 
$ brew install graphviz


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
