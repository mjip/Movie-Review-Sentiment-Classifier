#!/usr/bin/env python3

import nltk
import string
import codecs
import random
import re
import sklearn

from nltk.util import ngrams
from nltk.tokenize import RegexpTokenizer
from collections import Counter
from collections import defaultdict

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn import model_selection

##########################################
# pos data cleaning
##########################################

corpus_pos = []
stop_words = nltk.corpus.stopwords.words('english')
stemmer = nltk.stem.porter.PorterStemmer()

raw_pos = codecs.open("rt-polaritydata/rt-polarity.pos",encoding='utf-8',errors='ignore').readlines()
for line in raw_pos:
	tokenizer_pos = RegexpTokenizer(r'[\w]+')
	tokens_pos = tokenizer_pos.tokenize(line)
	start_tokens_pos = [w for w in tokens_pos if not w in stop_words]
	#start_tokens_pos = [stemmer.stem(w) for w in start_tokens_pos]
	corpus_pos.append(" ".join(start_tokens_pos))

##########################################
# neg data cleaning
##########################################

corpus_neg = []
raw_neg = codecs.open("rt-polaritydata/rt-polarity.neg",encoding='utf-8',errors='ignore').readlines()
for line in raw_neg:
	tokenizer_neg = RegexpTokenizer(r'[\w]+')
	tokens_neg = tokenizer_neg.tokenize(line)
	start_tokens_neg = [w for w in tokens_neg if not w in stop_words]
	#start_tokens_neg = [stemmer.stem(w) for w in start_tokens_neg]
	corpus_neg.append(" ".join(start_tokens_neg))

#########################################
# tagging & preparing test/training data
#########################################

neg_tag = ['neg'] * len(corpus_neg)
pos_tag = ['pos'] * len(corpus_pos)

corpus = corpus_neg + corpus_pos
tags = neg_tag + pos_tag 

# Removes accents off spanish reviews
cv = CountVectorizer(strip_accents='ascii')
bigram_cv = CountVectorizer(ngram_range=(2,2),strip_accents='ascii')
unigram_bigram_cv = CountVectorizer(ngram_range=(1,2),strip_accents='ascii')

corpus_cv = cv.fit_transform(corpus)
corpus_bigram_cv = bigram_cv.fit_transform(corpus)
corpus_unigram_bigram_cv = unigram_bigram_cv.fit_transform(corpus)

# Removes digits, keywords with digits from vector fit
cv.vocabulary_ = {k: v for k, v in cv.vocabulary_.items() if not k.isdigit() and not re.search(r'\d',k)}
bigram_cv.vocabulary_ = {k: v for k, v in cv.vocabulary_.items() if not k.isdigit() and not re.search(r'\d',k)}
unigram_bigram_cv = {k: v for k, v in cv.vocabulary_.items() if not k.isdigit() and not re.search(r'\d',k)}

#########################################
# training/classifying
#########################################

svc_model = svm.SVC(kernel='linear',C=1.0)
reg_model = linear_model.LogisticRegression()
bys_model = MultinomialNB()

rand_seed = 200
kf = KFold(n_splits=5,shuffle=True,random_state=rand_seed)
for m in [svc_model, reg_model, bys_model]:
	results = model_selection.cross_val_score(m, corpus_cv, tags, cv=kf)
	print("Accuracy: %.4f%% (%.4f%%)" % (results.mean()*100.0, results.std()*100.0))

c_train, c_test, t_train, t_test = train_test_split(corpus_cv, tags, test_size = 0.1, random_state = rand_seed)
c_train2, c_test2, t_train2, t_test2 = train_test_split(corpus_bigram_cv, tags, test_size = 0.1, random_state = rand_seed)
c_train3, c_test3, t_train3, t_test3 = train_test_split(corpus_unigram_bigram_cv, tags, test_size = 0.1, random_state = rand_seed)

print("Training set size: {}".format(len(t_train)))
print("Testing set size: {}".format(len(t_test)))

# setup random guesser
random.seed(rand_seed)
good_count = 0
for result in t_test:
	rand_num = random.randint(0,1)
	if round(rand_num) == 1:
		prediction = 'pos'
	else:
		prediction = 'neg'
	if(prediction == result):
		good_count = good_count + 1;

# training svm
svc = svm.SVC(kernel='linear', C = 1.0)
svc.fit(c_train, t_train);

svc2 = svm.SVC(kernel='linear', C = 1.0)
svc2.fit(c_train2, t_train2)

svc3 = svm.SVC(kernel='linear', C = 1.0)
svc3.fit(c_train3, t_train3)

# training naive bayes
nb = MultinomialNB();
nb.fit(c_train, t_train);

nb2 = MultinomialNB()
nb2.fit(c_train2, t_train2)

nb3 = MultinomialNB()
nb3.fit(c_train3, t_train3)

# training logistic regression
lr = linear_model.LogisticRegression();
lr.fit(c_train, t_train);

lr2 = linear_model.LogisticRegression();
lr2.fit(c_train2, t_train2)

lr3 = linear_model.LogisticRegression();
lr3.fit(c_train3, t_train3)

print("\nUnigram Results:")
print("Random Guessing: \t\t{}".format((float(good_count) / len(t_test))))
print("SVM (Linear): \t\t\t{}".format(svc.score(c_test, t_test)))
print("Multinomial Naive Bayes: \t{}".format(nb.score(c_test, t_test)))
print("Logistic Regression: \t\t{}\n".format(lr.score(c_test, t_test)))

print("Bigram Results:")
print("Random Guessing: \t\t{}".format((float(good_count) / len(t_test2))))
print("SVM (Linear): \t\t\t{}".format(svc2.score(c_test2, t_test2)))
print("Multinomial Naive Bayes: \t{}".format(nb2.score(c_test2, t_test2)))
print("Logistic Regression: \t\t{}\n".format(lr2.score(c_test2, t_test2)))

print("Unigram-Bigram Results:")
print("Random Guessing: \t\t{}".format((float(good_count) / len(t_test3))))
print("SVM (Linear): \t\t\t{}".format(svc3.score(c_test3, t_test3)))
print("Multinomial Naive Bayes: \t{}".format(nb3.score(c_test3, t_test3)))
print("Logistic Regression: \t\t{}\n".format(lr3.score(c_test3, t_test3)))

# Confusion matrix on best result
t_pred = nb3.predict(c_test3)

score = sklearn.metrics.accuracy_score(t_test3, t_pred)
conf_mat = sklearn.metrics.confusion_matrix(t_test3, t_pred, labels = ['neg', 'pos'])

print("Confusion Matrix Accuracy Score: {}".format(score))
print("Confusion Matrix: {}".format(conf_mat))
