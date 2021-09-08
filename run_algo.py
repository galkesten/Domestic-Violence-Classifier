import numpy as np
import pandas as pd
import csv
import nltk
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

results = []

def randonforest(x_train_new, Y, random_state_val):
    x_train_numpy = np.array(x_train_new)
    kf= StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state_val)
    accuracy = 0
    for train_index,test_index in kf.split(x_train_new, Y):
        X_train, X_test= x_train_numpy[train_index], x_train_numpy[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        # Create a Gaussian Classifier
        clf = RandomForestClassifier(n_estimators=100)
        # Train the model using the training sets y_pred=clf.predict(X_test)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy+= metrics.accuracy_score(y_test, y_pred)
    avg_acc = accuracy/5
    print(f"RF avg acurracy:{avg_acc} ")
    return avg_acc


def mlp(x_train_new, Y, random_state_val):
    x_train_numpy = np.array(x_train_new)
    kf= StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state_val)
    accuracy = 0
    for train_index,test_index in kf.split(x_train_new, Y):
        x_train, x_test= x_train_numpy[train_index], x_train_numpy[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        # mlp= MLPClassifier(max_iter=500, activation= 'relu', alpha= 0.0001, learning_rate='adaptive', solver= 'adam')
        mlp = MLPClassifier(max_iter=500, activation='relu')
        mlp.fit(x_train, y_train)
        y_pred = mlp.predict(x_test)
        a=metrics.accuracy_score(y_test, y_pred)
        accuracy+= a

    avg_acc = accuracy/5
    print(f"mlp avg acurracy:{avg_acc} ")
    return avg_acc



def svm_func(x_train_new, Y, random_state_val):
    x_train_numpy = np.array(x_train_new)
    kf= StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state_val)
    accuracy = 0
    for train_index,test_index in kf.split(x_train_new, Y):
        x_train, x_test= x_train_numpy[train_index], x_train_numpy[test_index]
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


def naiv_bayse(x_train_new, Y, random_state_val):
    x_train_numpy = np.array(x_train_new)
    kf= StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state_val)
    accuracy = 0
    for train_index,test_index in kf.split(x_train_new, Y):
        x_train, x_test= x_train_numpy[train_index], x_train_numpy[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        scalar=MinMaxScaler()
        x_train = scalar.fit_transform(x_train)
        x_test = scalar.fit_transform(x_test)
        cls=MultinomialNB()
        cls.fit(x_train, y_train)
        y_pred=cls.predict(x_test)
        a=metrics.accuracy_score(y_test, y_pred)
        accuracy+= a

    avg_acc = accuracy/5
    print(f"nb avg acurracy:{avg_acc} ")
    return avg_acc


def get_accuracy(x_train, Y, title):
    l = [title, 'RN', 'MLP', 'SVM', 'NB']
    i=42
    results.append(l)
    RN_acc = randonforest(x_train, Y, i)
    MLP_acc = mlp(x_train, Y, i)
    SVM_acc = svm_func(x_train, Y, i)
    NB_acc = naiv_bayse(x_train, Y, i)
    l = ['', RN_acc,MLP_acc, SVM_acc, NB_acc]
    results.append(l)

    with open('results.csv', 'w', newline='') as f:
        writer=csv.writer(f)
        writer.writerows(results)
