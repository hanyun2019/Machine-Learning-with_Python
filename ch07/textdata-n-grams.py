# Introduction to Machine Learning with Python
# Chapter 7: Working with Text Data
# Refreshed by Haowen Huang

from preamble import *

import mglearn
import numpy as np

from sklearn.datasets import load_files

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline


# In Chapter 5, we talked about two kinds of features that can represent properties of the data: 
# continuous features that describe a quantity, and categorical features that are items from a fixed list. 
# There is a third kind of feature that can be found in many application, which is text.

# There are four kinds of string data you might see:
# • Categorical data
# • Free strings that can be semantically mapped to categories(可以语义映射到类别的自由字符串)
# • Structured string data
# • Text data

# In the context of text analysis, the dataset if often called the corpus(语料库), 
# and each data point, represented as a single text, is called a document.

# These terms come from the information retrieval(检索) (IR) and natural language processing (NLP) community, 
# which both deal mostly in text data.

print("\n----------- Working with Text Data -----------")
# Example application: Sentiment analysis of movie reviews  影评的情感分析

reviews_train = load_files("/Users/research/MLPython/ch07/data/aclImdb/train/")
# load_files returns a bunch, containing training texts and training labels
text_train, y_train = reviews_train.data, reviews_train.target
print("type of text_train: {}".format(type(text_train)))
print("length of text_train: {}".format(len(text_train)))
print("text_train[6]:\n{}".format(text_train[6]))

# type of text_train: <class 'list'>
# length of text_train: 75000
# text_train[6]:
# b'Gloomy Sunday - Ein Lied von Liebe und Tod directed by Rolf Sch\xc3\xbcbel in 1999 is a romantic, absorbing, beautiful, and heartbreaking movie. It started like Jules and Jim; it ended as one of Agatha Christie\'s books, and in between it said something about love, friendship, devotion, jealousy, war, Holocaust, dignity, and betrayal, and it did better than The Black Book which is much more popular. It is not perfect, and it made me, a cynic, wonder in the end on the complexity of the relationships and sensational revelations, and who is who to whom but the movie simply overwhelmed me. Perfect or not, it is unforgettable. All four actors as the parts of the tragic not even a triangle but a rectangle were terrific. I do believe that three men could fell deeply for one girl as beautiful and dignified as Ilona in a star-making performance by young Hungarian actress Erica Marozs\xc3\xa1n and who would not? The titular song is haunting, sad, and beautiful, and no doubt deserves the movie been made about it and its effect on the countless listeners. I love the movie and I am surprised that it is so little known in this country. It is a gem.<br /><br />The fact that it is based on a story of the song that had played such important role in the lives of all characters made me do some research, and the real story behind the song of Love and Death seems as fascinating as the fictional one. The song was composed in 1930s by Rezs\xc3\xb6 Seress and was believed to have caused many suicides in Hungary and all over Europe as the world was moving toward the most devastating War of the last century. Rezs\xc3\xb6 Seress, a Jewish-Hungarian pianist and composer, was thrown to the Concentration Camp but survived, unlike his mother. In January, 1968, Seress committed suicide in Budapest by jumping out of a window. According to his obituary in the New York Times, "Mr. Seres complained that the success of "Gloomy Sunday" actually increased his unhappiness, because he knew he would never be able to write a second hit." <br /><br />Many singers from all over the world have recorded their versions of the songs in different languages. Over 70 performers have covered the song since 1935, and some famous names include Billie Holiday, Paul Robeson, Pyotr Leschenko (in Russian, under title "Mratschnoje Woskresenje"), Bjork, Sarah McLachlan, and many more. The one that really got to me and made me shiver is by Diamanda Gal\xc3\xa1s, the Greek born American singer/pianist/performer with the voice of such tragic power that I still can\'t get over her singing. Gal\xc3\xa1s has been described as "capable of the most unnerving vocal terror", and in her work she mostly concentrates on the topics of "suffering, despair, condemnation, injustice and loss of dignity." When she sings the Song of Love and Death, her voice that could\'ve belonged to the most tragic heroines of Ancient Greece leaves no hope and brings the horror and grief of love lost forever to the unbearable and incomparable heights.<br /><br />8.5/10'


# We can see that text_train is a list of length 75.000, where each entry is a string containing a review. We printed the review with index one. 
# You can see that the review contains some HTML line breaks ("<br />"). While these are unlikely to have a large impact on our machine learning models, 
# it is better to clean the data from this for‐ mating before we proceed:
text_train = [doc.replace(b"<br />", b" ") for doc in text_train]

print("\nnp.unique(y_train):\n",np.unique(y_train))
# np.unique(y_train):
#  [0 1 2]

print("\nSamples per class (training):\n {}".format(np.bincount(y_train)))
# Samples per class (training):
#  [12500 12500 50000]

reviews_test = load_files("/Users/research/MLPython/ch07/data/aclImdb/test/")
text_test, y_test = reviews_test.data, reviews_test.target
print("\nNumber of documents in test data: {}".format(len(text_test)))
print("\nSamples per class (test): {}".format(np.bincount(y_test)))
text_test = [doc.replace(b"<br />", b" ") for doc in text_test]
# Number of documents in test data: 25000
# Samples per class (test): [12500 12500]


# Let’s use the TfidfVectorizer on the IMDb movie review data and find the best setting of n-gram range using grid-search:
pipe = make_pipeline(TfidfVectorizer(min_df=5), LogisticRegression())
# running the grid-search takes a long time because of the
# relatively large grid and the inclusion of trigrams
param_grid = {'\nlogisticregression__C': [0.001, 0.01, 0.1, 1, 10, 100],
              "tfidfvectorizer__ngram_range": [(1, 1), (1, 2), (1, 3)]}

grid = GridSearchCV(pipe, param_grid, cv=5)
print("\nGridSearchCV(pipe, param_grid, cv=5):\n", grid)
# GridSearchCV(pipe, param_grid, cv=5):
#  GridSearchCV(cv=5, error_score='raise-deprecating',
#              estimator=Pipeline(memory=None,
#                                 steps=[('tfidfvectorizer',
#                                         TfidfVectorizer(analyzer='word',
#                                                         binary=False,
#                                                         decode_error='strict',
#                                                         dtype=<class 'numpy.float64'>,
#                                                         encoding='utf-8',
#                                                         input='content',
#                                                         lowercase=True,
#                                                         max_df=1.0,
#                                                         max_features=None,
#                                                         min_df=5,
#                                                         ngram_range=(1, 1),
#                                                         norm='l2',
#                                                         preprocessor=None,
#                                                         smooth_idf=True...
#                                                            multi_class='warn',
#                                                            n_jobs=None,
#                                                            penalty='l2',
#                                                            random_state=None,
#                                                            solver='warn',
#                                                            tol=0.0001,
#                                                            verbose=0,
#                                                            warm_start=False))],
#                                 verbose=False),
#              iid='warn', n_jobs=None,
#              param_grid={'\nlogisticregression__C': [0.001, 0.01, 0.1, 1, 10,
#                                                      100],
#                          'tfidfvectorizer__ngram_range': [(1, 1), (1, 2),
#                                                           (1, 3)]},
#              pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
#              scoring=None, verbose=0)

grid.fit(text_train, y_train)
print("\nBest cross-validation score: {:.2f}".format(grid.best_score_))
print("\nBest parameters:\n{}".format(grid.best_params_))

