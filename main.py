import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from PreProcessor import PreProcessor
from nltk.tokenize import TweetTokenizer
from sklearn import svm

df = pd.read_csv('vectors/USE/USE-Lemma.csv')
Y = np.array(df['Label'])
df = df.drop(columns=['Label'])
X = df.to_numpy()
X = np.delete(X, 0, 1)
tfidf = pd.read_csv("./vectors/tfIdf/tfidf.csv")
tfidf = tfidf.drop(columns=['Label'])
tfidf = tfidf.to_numpy()
tfidf = np.delete(tfidf, 0, 1)
X = np.hstack((X, tfidf))
preProcessor = PreProcessor(False, False)
preProcessor.splitDbToXandY()
X2,Y2  = preProcessor.X, preProcessor.Y
Y2 = np.array(Y2)
X2= np.array(X2)

accuracy = 0
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_index,test_index in kf.split(X, Y):
    X_train, X_test= X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    posts_test = X2[test_index]
    clf = svm.SVC()
    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    for i in range(0, len(y_pred)):
       if y_pred[i] != y_test[i]:
            print(f"post is {posts_test[i]}, real: {y_test[i]}, pred : {y_pred[i]}")
            print("........................................")
    accuracy+= metrics.accuracy_score(y_test, y_pred)



print (f"avg acurracy:{accuracy/5} ")

preProcessor = PreProcessor(False, False)
preProcessor.splitDbToXandY()
vectorizer = CountVectorizer()
vectors = vectorizer.fit_transform(preProcessor.X)
featurs1 = vectorizer.get_feature_names()
print(vectors.shape)
labels = preProcessor.Y

accuracy = 0
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_index,test_index in kf.split(X, Y):
    X_train, X_test= vectors[train_index], vectors[test_index]
    y_train, y_test =labels[train_index], labels[test_index]
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy+= metrics.accuracy_score(y_test, y_pred)
print (f"avg acurracy:{accuracy/5} ")


str = "hello my name is gal. i am gonna fight. I don't like you"
list1 = TweetTokenizer().tokenize(str)

print(list1)