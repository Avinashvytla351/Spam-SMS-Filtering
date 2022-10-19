import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import string
import seaborn as sns
import os
import math
import re
import nltk
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pickle
sms = pd.read_csv('D:\IARE\Project\smsSpamDetection\spam.csv',encoding='latin-1')
sms = sms.rename(columns={'v1':'label', 'v2':'text'})

# removing  useless columns
sms.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)

# Label mapping: ham->0, spam-> 1
sms['label_num'] = sms['label'].map({'ham':0, 'spam':1})

# Length column
sms['Length'] = sms['text'].apply(len)
def remove_punctuation(text):
    puncFree ="".join([i for i in text if i not in string.punctuation])
    return puncFree

sms['text'] = sms['text'].apply(lambda x: remove_punctuation(x))
# Lower Case
sms['text'] = sms['text'].apply(lambda x: x.lower())
sms['text'] = sms['text'].apply(lambda x: x.lower())


# dataframe -> array
X, y = np.asanyarray(sms['text']), np.asanyarray(sms['label_num'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=24)
len(X_train), len(X_test)
trainList=list(X_train)

counter_vec = CountVectorizer().fit(X_train)
X_train_vec, X_test_vec = counter_vec.transform(X_train), counter_vec.transform(X_test)

classifiers = [['Neural Network :', MLPClassifier(max_iter = 2000)]]

predictions_df = pd.DataFrame()
predictions_df['action'] = y_test

predictList = [trainList]
for name,classifier in classifiers:
    classifier = classifier
    predictList.append(classifier.fit(X_train_vec, y_train))
    predictions = classifier.predict(X_test_vec)
    predictions_df[name.strip(" :")] = predictions
    (name, accuracy_score(y_test, predictions))

pickle.dump(predictList,open("spam.pkl","wb"))