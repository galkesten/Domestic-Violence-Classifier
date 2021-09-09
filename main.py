import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from PreProcessor import PreProcessor

df = pd.read_csv('vectors/USE/USE-try-Lemma.csv')
Y = np.array(df['Label'])
df = df.drop(columns=['Label'])
X = df.to_numpy()
X = np.delete(X, 0, 1)

accuracy = 0
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_index,test_index in kf.split(X, Y):
    X_train, X_test= X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy+= metrics.accuracy_score(y_test, y_pred)



print (f"avg acurracy:{accuracy/5} ")

preProcessor = PreProcessor(False, False)
preProcessor.splitDbToXandY()
vectorizer = TfidfVectorizer(stop_words='english')
vectors = vectorizer.fit_transform(preProcessor.X)
#print(len(vectorizer.get_feature_names()))
#print(vectorizer.get_feature_names())
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