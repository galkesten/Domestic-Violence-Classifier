
from csv import DictWriter
from datetime import date
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB,MultinomialNB
import os


def getRandomForestAverageAccuracy(X, Y, random_state_val):
    """
    This function train a Random forest classifier with StratifiedKFold with K=5.
    :param X: The training input samples (suppose to be a feature matrix)
    :param Y: The target values (class labels in classification)
    :param random_state_val: random_state` affects the ordering of the indices, which controls the randomness of each fold for each class
    :return: average accuracy of all k folds.
    """
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state_val)
    accuracy = 0
    for train_index,test_index in kf.split(X, Y):
        X_train, X_test= X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        clf = RandomForestClassifier(n_estimators=100, random_state=random_state_val)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy+= metrics.accuracy_score(y_test, y_pred)
    avg_acc = accuracy/5
    return avg_acc


def getMLPAverageAccuracy(X, Y, random_state_val):
    """
        This function train a MLP classifier with StratifiedKFold with K=5.
        :param X: The training input samples (suppose to be a feature matrix)
        :param Y: The target values (class labels in classification)
        :param random_state_val: random_state` affects the ordering of the indices, which controls the randomness of each fold for each class
        :return: average accuracy of all k folds.
    """
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state_val)
    accuracy = 0
    for train_index,test_index in kf.split(X, Y):
        x_train, x_test= X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        mlp = MLPClassifier(max_iter=500, activation='relu', random_state=random_state_val)
        mlp.fit(x_train, y_train)
        y_pred = mlp.predict(x_test)
        a=metrics.accuracy_score(y_test, y_pred)
        accuracy += a

    avg_acc = accuracy/5
    return avg_acc


def getSVMAverageAccuracy(X, Y, random_state_val):
    """
        This function train a SVM classifier with StratifiedKFold with K=5.
        :param X: The training input samples (suppose to be a feature matrix)
        :param Y: The target values (class labels in classification)
        :param random_state_val: random_state` affects the ordering of the indices, which controls the randomness of each fold for each class
        :return: average accuracy of all k folds.
    """
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state_val)
    accuracy = 0
    for train_index,test_index in kf.split(X, Y):
        x_train, x_test= X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        cls=svm.SVC()
        cls.fit(x_train, y_train)
        y_pred=cls.predict(x_test)
        a = metrics.accuracy_score(y_test, y_pred)
        accuracy += a

    avg_acc = accuracy/5
    return avg_acc


def getNaiveBayesAverageAccuracy(X, Y, random_state_val, useMultinomial):
    """
     This function train a Naive Bayes classifier with StratifiedKFold with K=5.
     The Naive Bayes classifier will be Multinomial or Gaussian.
    :param X:  The training input samples (suppose to be a feature matrix)
    :param Y: The target values (class labels in classification)
    :param random_state_val: random_state` affects the ordering of the indices, which controls the randomness of each fold for each class
    :param useMultinomial: if true- a Multinomial naive bayes will be trained, else- Gaussian.
    :return: average accuracy of all k folds.
    """
    kf= StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state_val)
    accuracy = 0
    for train_index,test_index in kf.split(X, Y):
        x_train, x_test= X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        if useMultinomial:
            cls=MultinomialNB()
        else:
            cls = GaussianNB()
        cls.fit(x_train, y_train)
        y_pred=cls.predict(x_test)
        a=metrics.accuracy_score(y_test, y_pred)
        accuracy+= a

    avg_acc = accuracy/5
    return avg_acc


def calcAccuracy(X, Y, title, useRN=True, useMLP=True, useSvm=True, useNb=True, useMultinomialNB=False, numExperiment=1):
    """
    This function calculates the average accuracy of different classifiers that are trained on X featuers.
    The average accuracy for each classifier is than printed to a csv file.
    The name of the file will be in this format- Experiment{numExperiment}-results-{dateStr}.csv snd will be saved
    in Results directory.
    Each time this function called a new row will be added to the csv file and the accuracies will be printed
    in the appropriate column.
    :param X:  The training input samples (suppose to be a feature matrix)
    :param Y: The target values (class labels in classification)
    :param title: A string that will be printed in the title column when the new row will be added to the csv file.
    The string suppose to contain information about preprocessing method+ type of features.
    :param useRN: if true- a Random forest classifier will be trained with k cross validation an the accuracy score will printed
    to the "RD" colmn.
    :param useMLP:if true- a MLP classifier will be trained with k cross validation an the accuracy score will printed
    to the "MLP" column.
    :param useSvm: if true- a SVM classifier will be trained with k cross validation an the accuracy score will printed
    to the "SVM" colmn.
    :param useNb: if true- a Random forest classifier will be trained with k cross validation an the accuracy score will printed
    to the "NB" colmn.
    :param useMultinomialNB: if useNb is true, this bool variable will indicate whether or not to use Multinomial NB.
    :param numExperiment: should be 1/2/3. will determine the file name which the new results will be printed.
    :return: None
    """
    field_names = ['Title', 'RD', 'MLP', 'SVM', 'NB']
    random_state=42
    if useRN:
        RD_acc = getRandomForestAverageAccuracy(X, Y, random_state)
    else:
        RD_acc = -1
    if useMLP:
        MLP_acc = getMLPAverageAccuracy(X, Y, random_state)
    else:
        MLP_acc = -1
    if useSvm:
        SVM_acc = getSVMAverageAccuracy(X, Y, random_state)
    else:
        SVM_acc = -1
    if useNb:
        NB_acc = getNaiveBayesAverageAccuracy(X, Y, random_state, useMultinomialNB)
    else:
        NB_acc = -1
    dict = {'Title':title, 'RD': RD_acc, 'MLP': MLP_acc, 'SVM': SVM_acc, 'NB': NB_acc}
    today = date.today()
    dateStr = today.strftime("%b-%d-%Y")
    newfileName = f'./Results/Experiment{numExperiment}-results-{dateStr}.csv'
    fileExist = os.path.isfile(newfileName)
    with open(newfileName, 'a') as f_object:
        dictwriter_object = DictWriter(f_object, fieldnames=field_names)
        if not fileExist:
            dictwriter_object.writeheader()
        dictwriter_object.writerow(dict)
        f_object.close()
