import numpy as np
import pandas as pd
import csv
from csv import DictWriter
import nltk
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
import os

results = []

def randomforest(X, Y, random_state_val):
    #x_train_numpy = np.array(x_train_new)
    kf= StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state_val)
    accuracy = 0
    for train_index,test_index in kf.split(X, Y):
        X_train, X_test= X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        # Create a Gaussian Classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=random_state_val)
        # Train the model using the training sets y_pred=clf.predict(X_test)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy+= metrics.accuracy_score(y_test, y_pred)
    avg_acc = accuracy/5
    print(f"RF avg acurracy:{avg_acc} ")
    return avg_acc


def mlp(X, Y, random_state_val):
    #x_train_numpy = np.array(x_train_new)
    kf= StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state_val)
    accuracy = 0
    for train_index,test_index in kf.split(X, Y):
        x_train, x_test= X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        # mlp= MLPClassifier(max_iter=500, activation= 'relu', alpha= 0.0001, learning_rate='adaptive', solver= 'adam')
        mlp = MLPClassifier(max_iter=500, activation='relu', random_state=random_state_val)
        mlp.fit(x_train, y_train)
        y_pred = mlp.predict(x_test)
        a=metrics.accuracy_score(y_test, y_pred)
        accuracy+= a

    avg_acc = accuracy/5
    print(f"mlp avg acurracy:{avg_acc} ")
    return avg_acc



def svm_func(X, Y, random_state_val):
    #x_train_numpy = np.array(x_train_new)
    kf= StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state_val)
    accuracy = 0
    for train_index,test_index in kf.split(X, Y):
        x_train, x_test= X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        # cls=svm.SVC(C=16, kernel='linear', gamma='auto')
        cls=svm.SVC()
        cls.fit(x_train, y_train)
        y_pred=cls.predict(x_test)
        a=metrics.accuracy_score(y_test, y_pred)
        accuracy += a


    avg_acc = accuracy/5
    print(f"svm avg acurracy:{avg_acc} ")
    return avg_acc


def naiv_bayse(X, Y, random_state_val):
    #x_train_numpy = np.array(x_train_new)
    kf= StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state_val)
    accuracy = 0
    for train_index,test_index in kf.split(X, Y):
        x_train, x_test= X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        cls=GaussianNB()
        cls.fit(x_train, y_train)
        y_pred=cls.predict(x_test)
        a=metrics.accuracy_score(y_test, y_pred)
        accuracy+= a

    avg_acc = accuracy/5
    print(f"nb avg acurracy:{avg_acc} ")
    return avg_acc

def LR(X, Y, random_state_val):
    #x_train_numpy = np.array(x_train_new)
    kf= StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state_val)
    accuracy = 0
    for train_index,test_index in kf.split(X, Y):
        x_train, x_test= X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        cls=LogisticRegression(random_state=random_state_val)
        cls.fit(x_train, y_train)
        y_pred=cls.predict(x_test)
        a=metrics.accuracy_score(y_test, y_pred)
        accuracy += a

    avg_acc = accuracy/5
    print(f"LR avg acurracy:{avg_acc} ")
    return avg_acc


def get_accuracy(X, Y, title):
    field_names = ['Title', 'RN', 'MLP', 'SVM', 'NB']
    random_state=42
    RN_acc = randomforest(X, Y, random_state)
    MLP_acc = mlp(X, Y, random_state)
    SVM_acc = svm_func(X, Y, random_state)
    NB_acc = naiv_bayse(X, Y, random_state)
    dict = {'Title':title, 'RN': RN_acc, 'MLP': MLP_acc, 'SVM': SVM_acc, 'NB': NB_acc}
    # Open your CSV file in append mode
    # Create a file object for this file
    fileExist = os.path.isfile('./results.csv')
    with open('results.csv', 'a') as f_object:
        # Pass the file object and a list
        # of column names to DictWriter()
        # You will get a object of DictWriter
        dictwriter_object = DictWriter(f_object, fieldnames=field_names)
        if not fileExist:
            dictwriter_object.writeheader()
        # Pass the dictionary as an argument to the Writerow()
        dictwriter_object.writerow(dict)

        # Close the file object
        f_object.close()
