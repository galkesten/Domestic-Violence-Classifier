import pandas as pd
import numpy as np
from sklearn import svm
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from PreProcessor import PreProcessor
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow_hub as hub

"""
This script run the final classifier that wad described in our paper.
The script will ask you to type a post as an input, ant it will print its classification as an output.
"""

def runFinalClassifier():
	"""
	This function creates the final classifier that was described in our paper.
	The classifier is train on all the examples in the Domestic violence dataset.
	All the examples are preprocessed with the Minimal preprocessing method.
	Than a TFIDF vector is created for each  example. After the tfidf vector creation, The examples are e preprocessed
	with the Lemmatization preprocessing method. Then, the examples are sent to The universal Sentence Encoder which encodes
	each example to a semantic vector. The tfidf and the semantic vector are concatenate and a final vector representation
	is created for each example. Than final vectors are sent to SVM learning algorithm which creates our final classifier.
	Than, the function expects the user to enter as input a new post. A vector representation is created for the vector in the same
	way it was created for the training set. The vector is sent to the svm classifier and the classifier return its prediction.
	The function will than print to the screen the prediction- Critical/Not Critical.

	:return:
	"""
	df = pd.read_csv("vectors/USE/USE-Lemma.csv")
	Y = np.array(df['Label'])
	df = df.drop(columns=['Label'])
	X = df.to_numpy()
	X = np.delete(X, 0, 1)
	tfidf = pd.read_csv("vectors/TFIDF/tfidf.csv")
	tfidf = tfidf.drop(columns=['Label'])
	tfidf = tfidf.to_numpy()
	tfidf = np.delete(tfidf, 0, 1)
	X = np.hstack((X, tfidf))
	cls = svm.SVC()
	cls.fit(X,Y)

	preProcessor = PreProcessor()
	preProcessor.splitDbToXandY()
	vectorizer = TfidfVectorizer()
	X = vectorizer.fit(preProcessor.X)
	post = input("Enter a post which you want to classify")
	tokenized_post = word_tokenize(post.lower())
	minimal_preprocess_post = " ".join(tokenized_post)
	tfidf_vector_for_post = vectorizer.transform([minimal_preprocess_post])
	tfidf_vector_for_post = tfidf_vector_for_post.toarray()

	lemma_tokens = []
	wordnet_lemmatizer = WordNetLemmatizer()
	for word in tokenized_post:
		lemma = wordnet_lemmatizer.lemmatize(word)
		lemma_tokens.append(lemma)
	lemma_post=_post = " ".join(lemma_tokens)
	embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
	embeddings = embed([lemma_post])
	embeddings = [obj.numpy() for obj in embeddings]
	finalVector = np.hstack((embeddings, tfidf_vector_for_post))

	pred=cls.predict(finalVector)
	if pred == 1 :
		print("Critical")
	else:
		print("not Critical")

runFinalClassifier()