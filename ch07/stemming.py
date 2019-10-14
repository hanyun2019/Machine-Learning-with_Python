# Introduction to Machine Learning with Python
# Chapter 7: Working with Text Data: stemming
# Refreshed by Haowen Huang

from preamble import *

import mglearn
import numpy as np
import spacy
import nltk
import re

from sklearn.datasets import load_files

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedShuffleSplit


## 7.8 Advanced tokenization, stemming and lemmatization    高级分词、词干提取和词性还原
print("\n----------- Working with Text Data: Advanced tokenization, stemming and lemmatization -----------")

# load spacy's English-language models
# 安装 spacy 包之后需要下载相应的语言包：python -m spacy download en
en_nlp = spacy.load('en')
# instantiate nltk's Porter stemmer
stemmer = nltk.stem.PorterStemmer()

# define function to compare lemmatization in spacy with stemming in nltk
# 定义一个函数来对比 spacy 中的词形还原与 nltk 中的词干提取
def compare_normalization(doc):
    # tokenize document in spacy    在spacy中对文档进行分词
    doc_spacy = en_nlp(doc)
    # print lemmas found by spacy   spacy找到的词元
    print("Lemmatization:")
    print([token.lemma_ for token in doc_spacy])
    # print tokens found by Porter stemmer  词干提取器找到的词例
    print("Stemming:")
    print([stemmer.stem(token.norm_.lower()) for token in doc_spacy])

compare_normalization(u"Our meeting today was worse than yesterday, "
                       "I'm scared of meeting the clients tomorrow.")

# Lemmatization:
# ['-PRON-', 'meeting', 'today', 'be', 'bad', 'than', 'yesterday', ',', '-PRON-', 'be', 'scared', 'of', 'meet', 'the', 'client', 'tomorrow', '.']
# Stemming:
# ['our', 'meet', 'today', 'wa', 'wors', 'than', 'yesterday', ',', 'i', 'am', 'scare', 'of', 'meet', 'the', 'client', 'tomorrow', '.']

# Spacy 1.7.5 版将‘our’、'i'等代词全部还原为'-PRON-'，详情请参见spacy官方文档
# 想了解接口的细节，请参阅：nltk(http://www.nltk.org) 和 spacy(https://spacy.io/docs/)

# In general, lemmatization is a much more involved process than stemming, but usually produces better results when used for normalizing tokens for machine learning.
# 一般来说，词形还原是一个比词干提取更复杂的过程，但用于机器学习的词例标准化时通常可以给出比词干提取更好的结果。

# While scikit-learn implements neither form of normalization, 
# CountVectorizer allows specifying your own tokenizer to convert each document into a list of tokens using the tokenizer parameter. 
# CountVectorizer 允许使用 tokenizer 参数来指定使用你自己的分词器将每个文档转换为词例列表。

# We can use the lemmatization from Spacy to create a callable that will take a string and produce a list of lemmas:
# Technicallity: we want to use the regexp based tokenizer that is used by CountVectorizer 
# and only use the lemmatization from SpaCy. 
# To this end, we replace en_nlp.tokenizer (the SpaCy tokenizer) with the regexp based tokenization

# import re

# regexp used in CountVectorizer:
regexp = re.compile('(?u)\\b\\w\\w+\\b')
# load spacy language model
en_nlp = spacy.load('en', disable=['parser', 'ner'])
old_tokenizer = en_nlp.tokenizer
# replace the tokenizer with the preceding regexp
en_nlp.tokenizer = lambda string: old_tokenizer.tokens_from_list(
    regexp.findall(string))

# create a custom tokenizer using the SpaCy document processing pipeline
# (now using our own tokenizer)
def custom_tokenizer(document):
    doc_spacy = en_nlp(document)
    return [token.lemma_ for token in doc_spacy]

# define a count vectorizer with the custom tokenizer
lemma_vect = CountVectorizer(tokenizer=custom_tokenizer, min_df=5)



print("\n----------- Example application: Sentiment analysis of movie reviews -----------")
# Example application: Sentiment analysis of movie reviews  影评的情感分析

reviews_train = load_files("/Users/research/MLPython/ch07/data/aclImdb/train/")
# load_files returns a bunch, containing training texts and training labels
text_train, y_train = reviews_train.data, reviews_train.target
print("type of text_train: {}".format(type(text_train)))
print("length of text_train: {}".format(len(text_train)))
print("text_train[6]:\n{}".format(text_train[6]))

# Let’s transform the data and inspect the vocabulary size:
# transform text_train using CountVectorizer with lemmatization
X_train_lemma = lemma_vect.fit_transform(text_train)
print("\nX_train_lemma.shape: {}".format(X_train_lemma.shape))

# standard CountVectorizer for reference
vect = CountVectorizer(min_df=5).fit(text_train)
X_train = vect.transform(text_train)
print("\nX_train.shape: {}".format(X_train.shape))

# X_train_lemma.shape: (75000, 35168)
# X_train.shape: (75000, 44532)

# Lemmatization can be seen as a kind of regularization, as it conflates certain features. 
# Therefore, we expect lemmatization to improve performance most when the dataset is small. 

# To illustrate how lemmatization can help, we will use StratifiedShuffleSplit for cross-validation, 
# using only 1% of the data as training data, and the rest as test data:
# build a grid-search using only 1% of the data as training set:

# from sklearn.model_selection import StratifiedShuffleSplit

# using only 1% of the data as training data, and the rest as test data:
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.99,
                            train_size=0.01, random_state=0)
grid = GridSearchCV(LogisticRegression(), param_grid, cv=cv)
# perform grid search with standard CountVectorizer
grid.fit(X_train, y_train)
print("\nBest cross-validation score "
      "(standard CountVectorizer): {:.3f}".format(grid.best_score_))
# perform grid search with Lemmatization
grid.fit(X_train_lemma, y_train)
print("\nBest cross-validation score "
      "(lemmatization): {:.3f}".format(grid.best_score_))

# Best cross-validation score (standard CountVectorizer): 0.666
# Best cross-validation score (lemmatization): 0.666

# 可能这个数据集的数据量太大（75000条数据），因此 lemmatization(词形还原模型的效果不大) 