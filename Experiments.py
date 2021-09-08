import numpy as np
import pandas as pd
import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

df = pd.read_csv("db/DomecticViolence.csv")
X = np.array((df['Post']))
Y = np.array((df['Label']))

# Tokenization of each document
tokenized_doc = []
for d in X:
    tokenized_doc.append(word_tokenize(d.lower()))

# Convert tokenized document into gensim formated tagged data
tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_doc)]
## Train doc2vec model
#model = Doc2Vec(tagged_data, vector_size=300, min_count=2, epochs=40)
#model.build_vocab(tagged_data)
#model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
# Save trained doc2vec model
#model.save("test_doc2vec.model")
## Load saved doc2vec model
model= Doc2Vec.load("test_doc2vec.model")

x_train_new = []
for doc_id in range(len(tagged_data)):
    inferred_vector = model.infer_vector(tagged_data[doc_id].words)
    x_train_new.append(inferred_vector)

x_train_numpy = np.array(x_train_new)
kf= StratifiedKFold(n_splits=5, shuffle=True)
accuracy = 0
for train_index,test_index in kf.split(x_train_new, Y):
    X_train, X_test= x_train_numpy[train_index], x_train_numpy[test_index]
    posts,y_train, y_test = X[test_index], Y[train_index], Y[test_index]
    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=100)

    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # Model Accuracy, how often is the classifier correct?
   # for i in range(0, len(y_pred)):
    #    if y_pred[i]!= y_test[i]:
     #       print(f"post is {posts[i]}")
      #      print(f"y_pred is {y_pred[i]} and y_test is {y_test[i]}")
    accuracy+= metrics.accuracy_score(y_test, y_pred)
    print(metrics.f1_score(y_true=y_test, y_pred=y_pred))


print (f"avg acurracy:{accuracy/5} ")