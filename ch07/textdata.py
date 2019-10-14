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



## 7.3 Representing text data as Bag of Words   将文本数据表示为词袋
# Computing the bag-of-word representation for a corpus of documents consists of the following three steps:
# 1) Tokenization: Split each document into the words that appear in it (called tokens), for example by splitting them by whitespace(空格) and punctuation(标点).
# 2) Vocabulary building: Collect a vocabulary of all words that appear in any of the documents, and number them (say in alphabetical order).
# 3) Encoding: For each document, count how often each of the words in the vocabulary appear in this document.

## 7.3.1 Applying bag-of-words to a toy dataset
print("\n----------- Applying bag-of-words to a toy dataset -----------")

bards_words =["The fool doth think he is wise,",
              "but the wise man knows himself to be a fool"]

# from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()
vect.fit(bards_words)
print("\nCountVectorizer() configuration:\n", vect)
# CountVectorizer() configuration:
#  CountVectorizer(analyzer='word', binary=False, decode_error='strict',
#                 dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
#                 lowercase=True, max_df=1.0, max_features=None, min_df=1,
#                 ngram_range=(1, 1), preprocessor=None, stop_words=None,
#                 strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
#                 tokenizer=None, vocabulary=None)

print("\nVocabulary size: {}".format(len(vect.vocabulary_)))
print("\nVocabulary content:\n {}".format(vect.vocabulary_))
# Vocabulary size: 13
# Vocabulary content:
#  {'the': 9, 'fool': 3, 'doth': 2, 'think': 10, 'he': 4, 'is': 6, 'wise': 12, 'but': 1, 'man': 8, 'knows': 7, 'himself': 5, 'to': 11, 'be': 0}

bag_of_words = vect.transform(bards_words)
print("\nbag_of_words: {}".format(repr(bag_of_words)))
print("\nDense representation of bag_of_words:\n{}".format(bag_of_words.toarray()))
# bag_of_words: <2x13 sparse matrix of type '<class 'numpy.int64'>'
# 	with 16 stored elements in Compressed Sparse Row format>

# Dense representation of bag_of_words:
# [[0 0 1 1 1 0 1 0 0 1 1 0 1]
#  [1 1 0 1 0 1 0 1 1 1 0 1 1]]

# We can see that the word counts for each word are either zero or one, none of the two strings in bards_words contain a word twice. 
# You can read these feature vectors as follows: The first string "The fool doth think he is wise," is represented as the first row in, 
# and it contains the first word in the vocabulary, "be", zero times. It also contains the second word in the vocabulary, "but", zero times. 
# It does contain the third word, "doth", once, and so on. Looking at both rows, we can see that the fourth word, "fool", the tenth word "the" 
# and the thirteenth word "wise" appear in both strings.



## 7.3.2 Bag-of-word for movie reviews
print("\n----------- Applying bag-of-words to movie reviews dataset -----------")

vect = CountVectorizer().fit(text_train)
X_train = vect.transform(text_train)
print("\nX_train:\n{}".format(repr(X_train)))
# X_train:
# <75000x124255 sparse matrix of type '<class 'numpy.int64'>'
# 	with 10315542 stored elements in Compressed Sparse Row format>

# The shape of X_train, the bag-of-words representation of the training data, is 25.000 x 74.849, indicating that the vocabulary contains 74.849 entries. 
# Again, the data is stored as a SciPy sparse matrix. Let’s look in a bit more detail at the vocabulary. 
# Another way to access the vocabulary is using the get_feature_name method of the vectorizer, 
# which returns a convenient list where each entry corresponds to one feature:
feature_names = vect.get_feature_names()
print("\nNumber of features: {}".format(len(feature_names)))
print("\nFirst 20 features:\n{}".format(feature_names[:20]))
print("\nFeatures 20010 to 20030:\n{}".format(feature_names[20010:20030]))
print("\nEvery 2000th feature:\n{}".format(feature_names[::2000]))
# Number of features: 124255
# First 20 features:
# ['00', '000', '0000', '0000000000000000000000000000000001', '0000000000001', '000000001', '000000003', '00000001', '000001745', '00001', '0001', '00015', '0002', '0007', '00083', '000ft', '000s', '000th', '001', '002']

# Features 20010 to 20030:
# ['cheapen', 'cheapened', 'cheapening', 'cheapens', 'cheaper', 'cheapest', 'cheapie', 'cheapies', 'cheapjack', 'cheaply', 'cheapness', 'cheapo', 'cheapozoid', 'cheapquels', 'cheapskate', 'cheapskates', 'cheapy', 'chearator', 'cheat', 'cheata']

# Every 2000th feature:
# ['00', '_require_', 'aideed', 'announcement', 'asteroid', 'banquière', 'besieged', 'bollwood', 'btvs', 'carboni', 'chcialbym', 'clotheth', 'consecration', 'cringeful', 'deadness', 'devagan', 'doberman', 'duvall', 'endocrine', 'existent', 'fetiches', 'formatted', 'garard', 'godlie', 'gumshoe', 'heathen', 'honoré', 'immatured', 'interested', 'jewelry', 'kerchner', 'köln', 'leydon', 'lulu', 'mardjono', 'meistersinger', 'misspells', 'mumblecore', 'ngah', 'oedpius', 'overwhelmingly', 'penned', 'pleading', 'previlage', 'quashed', 'recreating', 'reverent', 'ruediger', 'sceme', 'settling', 'silveira', 'soderberghian', 'stagestruck', 'subprime', 'tabloids', 'themself', 'tpf', 'tyzack', 'unrestrained', 'videoed', 'weidler', 'worrisomely', 'zombified']


# from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import LogisticRegression
scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=5)
# scores = cross_val_score(LogisticRegression(solver='lbfgs',multi_class='auto',max_iter=5000), X_train, y_train, cv=5)
# 这一步的运行时间非常长！！！
print("\nMean cross-validation accuracy: {:.2f}".format(np.mean(scores)))
# Mean cross-validation accuracy: 0.71 (书上是：0.88)

print("\ny_train:",y_train)
# y_train: [2 2 2 ... 2 2 2]

# We obtain a mean cross-validation score of 71%, which indicates reasonable performance for a balanced binary classification task. 
# We know that LogisticRegression has a regularization parameter C which we can tune via cross-validation(交叉验证):

# from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)
print("\nBest cross-validation score: {:.2f}".format(grid.best_score_))
print("\nBest parameters: ", grid.best_params_)
# 这一步的运行时间非常长！！！
# Best cross-validation score: 0.72
# Best parameters:  {'C': 0.1}

# We obtain a cross-validation score of 88.8% using C=0.1. 
# We can now assess the generalization-performance of this parameter setting on the test set:
X_test = vect.transform(text_test)
print("\nTest score: {:.2f}".format(grid.score(X_test, y_test)))
# Test score: 0.88(书上的结果)
# Test score: 0.13(我跑出来的结果)

# Now, let’s see if we can improve the extraction of words. The way the CountVector izer extracts tokens is using a regular expression. 
# By default, the regular expression that is used is "\b\w\w+\b". 
# This means it finds all sequences of characters that consist of at least two letters or numbers ("\w") and that are separated by word boundaries ("\b"), 
# in particular it does not find single-letter words, and it splits up contractions like “doesn’t” or “bit.ly”, but matches “h8ter” as a single word. 
# The CountVectorizer then converts all words to lower-case characters, so that “soon”, “Soon” and “sOon” all correspond to the same token (and therefore feature).

# This simple mechanism works quite well in practice, but as we saw above, we get many uninformative features like the numbers. 
# One way to cut back on these is to only use tokens that appear in at least 2 documents (or at least 5 documents etc). 
# A token that appears only in a single document is unlikely to appear in the test set and is therefore not helpful.

# We can set the minimum number of documents a token needs to appear in with the min_df parameter:
vect = CountVectorizer(min_df=5).fit(text_train)
X_train = vect.transform(text_train)
print("\nX_train with min_df: {}".format(repr(X_train)))
# Result:
# 书上的结果：
# X_train with min_df: <25000x27271 sparse matrix of type '<class 'numpy.int64'>'
# 	with 3354014 stored elements in Compressed Sparse Row format>
# 我跑出来的结果：
# X_train with min_df: <75000x44532 sparse matrix of type '<class 'numpy.int64'>'
# 	with 10191240 stored elements in Compressed Sparse Row format>

# By requiring at least five appearances of each token, we can bring down the number of features to 27.272, 
# only about a third of the original features. Let’s look at some tokens again:
feature_names = vect.get_feature_names()

print("\nFirst 50 features:\n{}".format(feature_names[:50]))
print("\nFeatures 20010 to 20030:\n{}".format(feature_names[20010:20030]))
print("\nEvery 700th feature:\n{}".format(feature_names[::700]))
# 书上的结果：
# First 50 features:
# ['00', '000', '007', '00s', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '100', '1000', '100th', '101', '102', '103', '104', '105', '107', '108', '10s', '10th', '11', '110', '112', '116', '117', '11th', '12', '120', '12th', '13', '135', '13th', '14', '140', '14th', '15', '150', '15th', '16', '160', '1600', '16mm', '16s', '16th']
# Features 20010 to 20030:
# ['repentance', 'repercussions', 'repertoire', 'repetition', 'repetitions', 'repetitious', 'repetitive', 'rephrase', 'replace', 'replaced', 'replacement', 'replaces', 'replacing', 'replay', 'replayable', 'replayed', 'replaying', 'replays', 'replete', 'replica']
# Every 700th feature:
# ['00', 'affections', 'appropriately', 'barbra', 'blurbs', 'butchered', 'cheese', 'commitment', 'courts', 'deconstructed', 'disgraceful', 'dvds', 'eschews', 'fell', 'freezer', 'goriest', 'hauser', 'hungary', 'insinuate', 'juggle', 'leering', 'maelstrom', 'messiah', 'music', 'occasional', 'parking', 'pleasantville', 'pronunciation', 'recipient', 'reviews', 'sas', 'shea', 'sneers', 'steiger', 'swastika', 'thrusting', 'tvs', 'vampyre', 'westerns']
# 我跑出来的结果：
# First 50 features:
# ['00', '000', '001', '007', '00am', '00pm', '00s', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '100', '1000', '1001', '100k', '100th', '100x', '101', '101st', '102', '103', '104', '105', '106', '107', '108', '109', '10am', '10pm', '10s', '10th', '10x', '11', '110', '1100', '110th', '111', '112', '1138', '115', '116', '117', '11pm', '11th']
# Features 20010 to 20030:
# ['inert', 'inertia', 'inescapable', 'inescapably', 'inevitability', 'inevitable', 'inevitably', 'inexcusable', 'inexcusably', 'inexhaustible', 'inexistent', 'inexorable', 'inexorably', 'inexpensive', 'inexperience', 'inexperienced', 'inexplicable', 'inexplicably', 'inexpressive', 'inextricably']
# Every 700th feature:
# ['00', 'accountability', 'alienate', 'appetite', 'austen', 'battleground', 'bitten', 'bowel', 'burton', 'cat', 'choreographing', 'collide', 'constipation', 'creatively', 'dashes', 'descended', 'dishing', 'dramatist', 'ejaculation', 'epitomize', 'extinguished', 'figment', 'forgot', 'garnished', 'goofy', 'gw', 'hedy', 'hormones', 'imperfect', 'insomniac', 'janitorial', 'keira', 'lansing', 'linfield', 'mackendrick', 'masterworks', 'miao', 'moorehead', 'natassia', 'nude', 'ott', 'particulars', 'phillipines', 'pop', 'profusely', 'raccoons', 'redolent', 'responding', 'ronno', 'satirist', 'seminal', 'shrews', 'smashed', 'spendthrift', 'stocked', 'superman', 'tashman', 'tickets', 'travelling', 'uncomfortable', 'uprising', 'vivant', 'whine', 'x2']

# There are clearly much fewer numbers, and some of the more obscure words or misspellings seem to have vanished. 
# Let’s see how well our model performs by doing a grid-search again:
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)
print("\nBest cross-validation score: {:.2f}".format(grid.best_score_))
# Best cross-validation score: 0.72

# The best validation accuracy of the grid-search is still 72%, unchanged from before. 
# We didn’t improve our model, but having less features to deal with speeds up processing 
# and throwing away useless features might make the model more interpretable.


## 7.4 Stop-words
# Another way that we can get rid of uninformative words is by discarding words that are too frequent to be informative. 
# There are two main approaches: using a language-specific list of stop words, or discarding words that appear too frequently. 
# Scikit-learn had a built-in list of English stop-words in the feature_extraction.text module:
print("\n----------- Stop-words -----------")

# from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

print("\nNumber of stop words: {}".format(len(ENGLISH_STOP_WORDS)))
print("\nEvery 10th stopword:\n{}".format(list(ENGLISH_STOP_WORDS)[::10]))
# Number of stop words: 318
# Every 10th stopword:
# ['will', 'amoungst', 'become', 'therefore', 'latterly', 'get', 'whither', 'whenever', 'thick', 'put', 'only', 'too', 'out', 'front', 'neither', 'alone', 'done', 'name', 'seeming', 'amount', 'the', 'than', 'whose', 'via', 'forty', 'thin', 'nothing', 'should', 'is', 'almost', 'along', 'because']

# Specifying stop_words="english" uses the built-in list.
# We could also augment it and pass our own.
vect = CountVectorizer(min_df=5, stop_words="english").fit(text_train)
X_train = vect.transform(text_train)
print("\nX_train with stop words:\n{}".format(repr(X_train)))
# X_train with stop words:
# <75000x44223 sparse matrix of type '<class 'numpy.int64'>'
# 	with 6577418 stored elements in Compressed Sparse Row format>

# There are now 309 (= 44532 - 44223) less features in the dataset, which means that most, but not all of the stop-words appeared. 
# Let’s run the grid-search again:

grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)
print("\nBest cross-validation score: {:.2f}".format(grid.best_score_))
# Best cross-validation score: 0.88(书上的结果)
# Best cross-validation score: 0.72(我跑出来的结果)

# The grid-search performance decreased slightly using the stop words. 
# The change is very slight, but given that excluding 309 features is unlikely to change performance or interpretability a lot, 
# it doesn’t seem worth using this list. 

# As an exercise, you can try out the other approach, discarding frequently appearing words, 
# by setting the max_df option of CountVectorizer and see how it influences the number of features and the performance.




## 7.5 Rescaling the Data with tf-idf   用tf-idf缩放数据 (tf-idf: term frequency-inverse document frequency)
# Instead of dropping features that are deemed unimportant, another approach is to rescale features by how informative we expect them to be. 
# One of the most common ways to do this is using the term frequency–inverse document frequency (tf-idf) method. 

# Scikit-learn implements the tf-idf method in two classes, 
# 1) TfidfTransformer: takes in the sparse matrix output produced by CountVectorizer and transforms it
# 2) TfidfVectorizer: takes in the text data and does both the bag-of-words feature extraction and the tf-idf transformation
# The wikipedia page of tf-idf: https://en.wikipedia.org/wiki/Tf-idf

# 假如一篇文件的总词语数是100个，而词语“母牛”出现了3次，那么“母牛”一词在该文件中的词频(tf)就是3/100=0.03。
# 计算文件频率 (DF) 的方法是测定有多少份文件出现过“母牛”一词，然后除以文件集里包含的文件总数;
# 所以如果“母牛”一词在1,000份文件出现过，而文件总数是10,000,000份的话，其逆向文件频率(idf)就是 ln(10,000,000 / 1,000)=4。
# 最后的TF-IDF的分数为0.03 * 4=0.12。
print("\n----------- Rescaling the Data with tf-idf -----------")

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.pipeline import make_pipeline

pipe = make_pipeline(TfidfVectorizer(min_df=5, norm=None), LogisticRegression())
param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10]}

grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(text_train, y_train)
print("\nBest cross-validation score: {:.2f}".format(grid.best_score_))
# Best cross-validation score: 0.72

# As you can see, there is some improvement of using tf-idf instead of using just word counts. We can also inspect which words tf-idf found most important. 
# Keep in mind that the tf-idf scaling is meant to find words that distinguish documents, but it is a purely unsupervised technique. 
# So “important” here does not necessarily related to the “positive review” and “negative review” labels we are interested in. 
# First we extract the TfidfVectorizer from the pipeline:
vectorizer = grid.best_estimator_.named_steps["tfidfvectorizer"]
# transform the training dataset:
X_train = vectorizer.transform(text_train)
# find maximum value for each of the features over dataset:
max_value = X_train.max(axis=0).toarray().ravel()
sorted_by_tfidf = max_value.argsort()
# get feature names
feature_names = np.array(vectorizer.get_feature_names())
print("\nFeatures with lowest tfidf:\n{}".format(feature_names[sorted_by_tfidf[:20]]))
print("\nFeatures with highest tfidf: \n{}".format(feature_names[sorted_by_tfidf[-20:]]))
# Features with lowest tfidf:
# ['remained' 'acclaimed' 'combines' 'rapidly' 'uniformly' 'diverse'
#  'avoiding' 'fills' 'feeble' 'admired' 'wherever' 'admission' 'abound'
#  'starters' 'assure' 'pivotal' 'comprehend' 'deliciously' 'strung'
#  'inadvertently']
# Features with highest tfidf: 
# ['nukie' 'reno' 'dominick' 'taz' 'ling' 'rob' 'victoria' 'turtles'
#  'khouri' 'lorenzo' 'id' 'zizek' 'elwood' 'nikita' 'rishi' 'timon'
#  'titanic' 'zohan' 'pammy' 'godzilla']

# Features with low tf-idf are those that are either very commonly used across documents, or are only used sparingly, and only in very long documents. 
# Interestingly, many of the high tf-idf features actually identify certain shows or movies. 

# We can also find the words that have low inverse document frequency, that is those that appear frequently and are therefore deemed less important. 
# The inverse document frequency values found on the training set are stored in the idf_ attribute:
sorted_by_idf = np.argsort(vectorizer.idf_)
print("\nFeatures with lowest idf:\n{}".format(feature_names[sorted_by_idf[:100]]))
# Features with lowest idf:
# ['the' 'and' 'of' 'to' 'this' 'is' 'it' 'in' 'that' 'but' 'for' 'with'
#  'was' 'as' 'on' 'movie' 'not' 'one' 'be' 'have' 'are' 'film' 'you' 'all'
#  'at' 'an' 'by' 'from' 'so' 'like' 'who' 'there' 'they' 'his' 'if' 'out'
#  'just' 'about' 'he' 'or' 'has' 'what' 'some' 'can' 'good' 'when' 'more'
#  'up' 'time' 'very' 'even' 'only' 'no' 'see' 'would' 'my' 'story' 'really'
#  'which' 'well' 'had' 'me' 'than' 'their' 'much' 'were' 'get' 'other' 'do'
#  'been' 'most' 'also' 'into' 'don' 'her' 'first' 'great' 'how' 'made'
#  'people' 'will' 'make' 'because' 'way' 'could' 'bad' 'we' 'after' 'them'
#  'too' 'any' 'then' 'movies' 'watch' 'she' 'think' 'seen' 'acting' 'its'
#  'characters']

# As expected, these are mostly English stop words like “the” and “no”. 



## 7.6 Investigating model coefficients 研究模型系数



## 7.7 Bag of words with more than one word (n-grams)   多个单词的词袋(n元分词)
# One of the main disadvantages of using a bag-of-word representation is that word order is completely discarded. 
# There is a way of capturing context when using a bag-of-word representation, by not only considering the counts of single tokens, 
# but also the counts of pairs or triples of tokens that appear next to each other.
print("\n----------- Bag of words with more than one word (n-grams) -----------")

print("bards_words:\n{}".format(bards_words))
# bards_words:
# ['The fool doth think he is wise,', 'but the wise man knows himself to be a fool']

# The default is to create one feature per sequence of tokens that are at least one token long, and at most one token long, 
# in other words exactly one token long (single tokens are also called unigrams):
cv = CountVectorizer(ngram_range=(1, 1)).fit(bards_words)
print("\nVocabulary size: {}".format(len(cv.vocabulary_)))
print("\nVocabulary:\n{}".format(cv.get_feature_names()))
print("\nVocabulary content:\n{}".format(cv.vocabulary_))
# Vocabulary size: 13
# Vocabulary:
# ['be', 'but', 'doth', 'fool', 'he', 'himself', 'is', 'knows', 'man', 'the', 'think', 'to', 'wise']
# Vocabulary content:
# {'the': 9, 'fool': 3, 'doth': 2, 'think': 10, 'he': 4, 'is': 6, 'wise': 12, 'but': 1, 'man': 8, 'knows': 7, 'himself': 5, 'to': 11, 'be': 0}
print("\nTransformed data (dense):\n{}".format(cv.transform(bards_words).toarray()))
# Transformed data (dense):
# [[0 0 1 1 1 0 1 0 0 1 1 0 1]
#  [1 1 0 1 0 1 0 1 1 1 0 1 1]]


# To look only at bigrams, that is only at sequences of two tokens following each other, we can set ngram_range to (2, 2):
cv = CountVectorizer(ngram_range=(2, 2)).fit(bards_words)
print("\nVocabulary size: {}".format(len(cv.vocabulary_)))
print("\nVocabulary:\n{}".format(cv.get_feature_names()))
print("\nVocabulary content:\n{}".format(cv.vocabulary_))
# Vocabulary size: 14
# Vocabulary:
# ['be fool', 'but the', 'doth think', 'fool doth', 'he is', 'himself to', 'is wise', 'knows himself', 'man knows', 'the fool', 'the wise', 'think he', 'to be', 'wise man']
# Vocabulary content:
# {'the fool': 9, 'fool doth': 3, 'doth think': 2, 'think he': 11, 'he is': 4, 'is wise': 6, 'but the': 1, 'the wise': 10, 'wise man': 13, 'man knows': 8, 'knows himself': 7, 'himself to': 5, 'to be': 12, 'be fool': 0}
print("\nTransformed data (dense):\n{}".format(cv.transform(bards_words).toarray()))
# Transformed data (dense). There is no common bigram between the two phrases in bard_words:
# [[0 0 1 1 1 0 1 0 0 1 0 1 0 0]
#  [1 1 0 0 0 1 0 1 1 0 1 0 1 1]]


cv = CountVectorizer(ngram_range=(1, 3)).fit(bards_words)
print("\nVocabulary size: {}".format(len(cv.vocabulary_)))
print("\nVocabulary:\n{}".format(cv.get_feature_names()))
print("\nVocabulary content:\n{}".format(cv.vocabulary_))
# Vocabulary size: 39
# Vocabulary:
# ['be', 'be fool', 'but', 'but the', 'but the wise', 'doth', 'doth think', 'doth think he', 'fool', 'fool doth', 'fool doth think', 'he', 'he is', 'he is wise', 'himself', 'himself to', 'himself to be', 'is', 'is wise', 'knows', 'knows himself', 'knows himself to', 'man', 'man knows', 'man knows himself', 'the', 'the fool', 'the fool doth', 'the wise', 'the wise man', 'think', 'think he', 'think he is', 'to', 'to be', 'to be fool', 'wise', 'wise man', 'wise man knows']
# Vocabulary content:
# {'the': 25, 'fool': 8, 'doth': 5, 'think': 30, 'he': 11, 'is': 17, 'wise': 36, 'the fool': 26, 'fool doth': 9, 'doth think': 6, 'think he': 31, 'he is': 12, 'is wise': 18, 'the fool doth': 27, 'fool doth think': 10, 'doth think he': 7, 'think he is': 32, 'he is wise': 13, 'but': 2, 'man': 22, 'knows': 19, 'himself': 14, 'to': 33, 'be': 0, 'but the': 3, 'the wise': 28, 'wise man': 37, 'man knows': 23, 'knows himself': 20, 'himself to': 15, 'to be': 34, 'be fool': 1, 'but the wise': 4, 'the wise man': 29, 'wise man knows': 38, 'man knows himself': 24, 'knows himself to': 21, 'himself to be': 16, 'to be fool': 35}
# Transformed data (dense):
# [[0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 1 1 0 0 0 0 0 0 1 1 1 0 0 1 1 1 0 0 0
#   1 0 0]
#  [1 1 1 1 1 0 0 0 1 0 0 0 0 0 1 1 1 0 0 1 1 1 1 1 1 1 0 0 1 1 0 0 0 1 1 1
#   1 1 1]]


# Let’s use the TfidfVectorizer on the IMDb movie review data and find the best setting of n-gram range using grid-search:
pipe = make_pipeline(TfidfVectorizer(min_df=5), LogisticRegression())
# running the grid-search takes a long time because of the
# relatively large grid and the inclusion of trigrams
param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10, 100],
              "tfidfvectorizer__ngram_range": [(1, 1), (1, 2), (1, 3)]}

grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(text_train, y_train)
print("\nBest cross-validation score: {:.2f}".format(grid.best_score_))
print("\nBest parameters:\n{}".format(grid.best_params_))



## 7.8 Advanced tokenization, stemming and lemmatization    高级分词、词干提取和词性还原
# Please refer to the following python program about 'Advanced tokenization, stemming and lemmatization'
# stemming.py


## 7.9 Topic Modeling and Document Clustering   主题建模和文档聚类
# Please refer to the following python program about 'Topic Modeling and Document Clustering'
# topicmodeling.py


## 7.10 Summary and Outlook
# In particular for text classification such as spam and fraud detection or sentiment analysis, 
# bag of word representations provide a simple and powerful solution. 
# 
# As so often in machine learning, the representation of the data is key in NLP applications, 
# and inspecting the tokens and n-grams that are extracted can give powerful insights into the modeling process. 
# 
# In text processing applications, it is often possible to introspect models in a meaningful way, 
# as we saw above, both for supervised and unsupervised tasks. 
# 
# 
# There have been several very exciting new developments in text processing in recent years.
# 1) The first is the use of continuous vector representations, also known as word vectors or distributed word representations, 
# as implemented in the word2vec library. The original paper “Distributed representations of words and phrases and their compositionality” 
# by Mikolov, Suskever, Chen, Corrado and Dean is a great introduction to the subject. 
# 2) Another direction in NLP that has picked up momentum in recent years are recurrent neural networks (RNNs) for text processing.