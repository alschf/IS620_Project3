# IS620 - Project 3
# Program: project3rev2.py
# Student: Neil Acampa
# Date:    10/24/16
# Function:



# 1.Using any of the three classifiers described in chapter 6 of Natural Language Processing with Python,
#   and any features you can think of, build the best name gender classifier you can. Begin by splitting the
#   Names Corpus into three subsets: 500 words for the test set, 500 words for the dev-test set, and the
#   remaining 6900 words for the training set. Then, starting with the example name gender classifier,
#   make incremental improvements. Use the dev-test set to check your progress. Once you are satisfied
#   with your classifier, check its final performance on the test set. How does the performance on the test
#   set compare to the performance on the dev-test set? Is this what you'd expect?
#   Source: Natural Language Processing with Python, exercise 6.10.2.



from __future__ import absolute_import 
from __future__ import division
import re
import os 
import math
import decimal
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import networkx as nx
import random
from urllib import urlopen
import nltk
nltk.download('gutenberg')
from nltk import word_tokenize
nltk.download('maxent_treebank_pos_tagger')
nltk.download('punkt')
tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')
#nltk.download('names')
from nltk.corpus import names





linelst=[]
lines  = ""
allwords       = []   # Contains all words
masterdict     = []   # Contains unique words
masterdictcnt  = []   # Contains count of unique words corresponding to masterdict
uniquewords    = []   # Contains unique words in the first dimension and the count in the second dim
                      # Will try to use if it works for unique word count

vocab          = []   # Unique words sorted by descending count
vocab200       = []   # Top 200 vocabulary words

vocabF         = []   # Vocabulary words for Freq Dist


# Table Elements

fheadings      = []
fheadings.append("Last Letter")
fheadings.append("Last Letter remove special chars")

fheadings.append("First Letter")
fheadings.append("First and Last Letter")
fheadings.append("First, Middle and Last Letter")



rejectchars = [',','.','?','<','>','!','"','-','%','&','#','(',')','*',';'];
rcnt = len(rejectchars);


def gender_features(name): 
  """Test"""
  return {'lastletter': name[-1]}
 

def gender_features1(name):
 return {'lastletter': name[0]}

def gender_features2(name):
  features = {}
  features['firstletter'] =  name[0].lower()
  features['lastletter']  =  name[-1].lower()
  return features


def gender_features3(name):
  m=int(round(len(name)/2))
  midletter = name[m].lower()
  features = {}
  features['first_letter']  =  name[0].lower()
  features['middle_letter'] =  midletter 
  features['last_letter']   =  name[-1].lower()
  return features



results = []
names = ([(name, 'male') for name in names.words('male.txt')] + [(names, 'female') for name in names.words('female.txt')])
random.shuffle(names)



train_names  = names[1500:]
devtest_names = names[500:1500]
test_names   = name[:500]


# gender_features = Last Letter
train_set   = [(gender_features(n), g) for (n,g) in train_names]
devtest_set = [(gender_features(n), g) for (n,g) in devtest_names]
test_set    = [(gender_features(n), g) for (n,g) in test_names]

classifier = nltk.NaiveBayesClassifier.train(train_set)
accuracy   = nltk.classify.accuracy(classifier, test_set)

results.append(accuracy)


# gender_features = First Letter
train_set   = [(gender_features1(n), g) for (n,g) in train_names]
devtest_set = [(gender_features1(n), g) for (n,g) in devtest_names]
test_set    = [(gender_features1(n), g) for (n,g) in test_names]

classifier = nltk.NaiveBayesClassifier.train(train_set)
accuracy   = nltk.classify.accuracy(classifier, test_set)

results.append(accuracy)

# gender_features = First and Last Letter
train_set   = [(gender_features2(n), g) for (n,g) in train_names]
devtest_set = [(gender_features2(n), g) for (n,g) in devtest_names]
test_set    = [(gender_features2(n), g) for (n,g) in test_names]

classifier = nltk.NaiveBayesClassifier.train(train_set)
accuracy   = nltk.classify.accuracy(classifier, test_set)

results.append(accuracy)

# gender_features = First, Middle Leter and  Last Letter
train_set   = [(gender_features3(n), g) for (n,g) in train_names]
devtest_set = [(gender_features3(n), g) for (n,g) in devtest_names]
test_set    = [(gender_features3(n), g) for (n,g) in test_names]

classifier = nltk.NaiveBayesClassifier.train(train_set)
accuracy   = nltk.classify.accuracy(classifier, test_set)

results.append(accuracy)

print ("%s\t%s\t%s") % ("Feature" , "Feature Desc", "Accuracy")
indx = 0
for i in range(4):
   indx = indx + 1
   print("%d\t%s\t%.4f") % (indx, fheading[i], results[1])
  



 