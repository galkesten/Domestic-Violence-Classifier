from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import best_params
from PreProcessor import PreProcessor
from run_algo import get_accuracy
import run_algo
import pandas as pd
import numpy as np
import spacy
import os
from csv import DictWriter

def createBagOfPOSVectors():
	nlp = spacy.load('en_core_web_sm')
	preProcessor = PreProcessor(removeStopWords=False, useLemmatization=False)
	preProcessor.splitDbToXandY()
	X = preProcessor.X
	docs_list = []
	for post in X:
		doc = nlp(post)
		pos_array = doc.to_array("POS")
		templist = [str(item) for item in pos_array]
		docs_list.append(" ".join(templist))
	vectorizer = CountVectorizer()
	X = vectorizer.fit_transform(docs_list)
	X = X.toarray()
	Y = np.array(preProcessor.Y)
	return X, Y

def createPOSWithTfidfVectors():
	nlp = spacy.load('en_core_web_sm')
	preProcessor = PreProcessor(removeStopWords=False, useLemmatization=False)
	preProcessor.splitDbToXandY()
	X = preProcessor.X
	docs_list = []
	for post in X:
		doc = nlp(post)
		pos_array = doc.to_array("POS")
		templist = [str(item) for item in pos_array]
		docs_list.append(" ".join(templist))
	vectorizer = TfidfVectorizer()
	X = vectorizer.fit_transform(docs_list)
	X = X.toarray()
	Y = np.array(preProcessor.Y)
	return X, Y

def getXandY(filePath, useTfidf=False, useBagOfWords=False, usePOSWithTfidf = False, useBagOfPOS= False):
	df = pd.read_csv(filePath)
	Y = np.array(df['Label'])
	df = df.drop(columns=['Label'])
	X = df.to_numpy()
	X = np.delete(X, 0, 1)
	if useTfidf:
		tfidf = pd.read_csv("./vectors/tfIdf/tfidf.csv")
		tfidf = tfidf.drop(columns=['Label'])
		tfidf = tfidf.to_numpy()
		tfidf = np.delete(tfidf, 0, 1)
		X = np.hstack((X, tfidf))
	if useBagOfWords:
		bagOfWords = pd.read_csv("./vectors/bagOfWords/BagOfWords.csv")
		bagOfWords = bagOfWords.drop(columns=['Label'])
		bagOfWords = bagOfWords.to_numpy()
		bagOfWords = np.delete(bagOfWords, 0, 1)
		X = np.hstack((X, bagOfWords))
	if usePOSWithTfidf:
		POSwithTfidf = pd.read_csv("./vectors/POSWithTfidf/POSWithTfidf.csv")
		POSwithTfidf = POSwithTfidf.drop(columns=['Label'])
		POSwithTfidf = POSwithTfidf.to_numpy()
		POSwithTfidf = np.delete(POSwithTfidf, 0, 1)
		X = np.hstack((X, POSwithTfidf))
	if useBagOfPOS:
		bagOfPOS = pd.read_csv("./vectors/bagOfPOS/BagOfPOS.csv")
		bagOfPOS = bagOfPOS.drop(columns=['Label'])
		bagOfPOS = bagOfPOS.to_numpy()
		bagOfPOS = np.delete(bagOfPOS, 0, 1)
		X = np.hstack((X, bagOfPOS))
	return X,Y

def bagOfWordsExperiment(useTfidf = False, useBagOfPOS=False, usePOSWithTfidf=False, stopWordsRemoval=True, useNb=True):
	#no preprocess
	header = "bagOfWords"
	if useTfidf:
		header+= "+tfidf"
	if useBagOfPOS:
		header+="+bagOfPos"
	if usePOSWithTfidf:
		header+="+POSWithTfidf"
	X,Y = getXandY(filePath="./vectors/bagOfWords/BagOfWords.csv", useTfidf=useTfidf, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
	get_accuracy(X,Y, header+ "-min preprocess", useNb=useNb)

	if stopWordsRemoval:
		#with stop words removal
		X, Y = getXandY(filePath="./vectors/bagOfWords/BagOfWords-stopWords.csv", useTfidf=useTfidf, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
		get_accuracy(X, Y, header+ "-stop words removal", useNb=useNb)

	#with Lemmatization
	X,Y =  getXandY(filePath="./vectors/bagOfWords/BagOfWords-Lemma.csv", useTfidf=useTfidf, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
	get_accuracy(X, Y, header+ "-Lemmatization", useNb=useNb)

	if stopWordsRemoval:
		#with stop words removal and Lemmatization
		X, Y = getXandY(filePath="./vectors/bagOfWords/BagOfWords-stopWords-Lemma.csv", useTfidf=useTfidf, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
		get_accuracy(X, Y, header+"-Lemmatization + stop words removal", useNb=useNb)

def tfidfExperiment(useBagOfPOS=False, usePOSWithTfidf=False, stopWordsRemoval=True, useNb=True):
	#no preprocess
	header = "tfidf"
	if useBagOfPOS:
		header+="+bagOfPos"
	if usePOSWithTfidf:
		header+="+POSWithTfidf"

	X,Y = getXandY(filePath="./vectors/tfIdf/tfidf.csv", useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
	get_accuracy(X,Y, header+ "-min preprocess", useNb=useNb)
	if stopWordsRemoval:
		#with stop words removal
		X, Y = getXandY(filePath="./vectors/tfIdf/tfidf-stopWords.csv", useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
		get_accuracy(X, Y, header+ "-stop words removal", useNb=useNb)

	#with Lemmatization
	X,Y =  getXandY(filePath="./vectors/tfIdf/tfidf-Lemma.csv", useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
	get_accuracy(X, Y, header+ "-Lemmatization", useNb=useNb)

	if stopWordsRemoval:
		#with stop words removal and Lemmatization
		X, Y = getXandY(filePath="./vectors/tfIdf/tfidf-stopWords-Lemma.csv", useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
		get_accuracy(X, Y, header+"-Lemmatization + stop words removal", useNb=useNb)

def BagOfWordsWithTfIdfExperiement():
	header = "bagOfWords+tfidf"
	X1, Y = getXandY("./vectors/bagOfWords/BagOfWords.csv")
	X2, _ = getXandY("./vectors/tfIdf/tfidf.csv")
	X = np.hstack((X1, X2))
	get_accuracy(X, Y, header + "-min preprocess")
	X1, Y = getXandY("./vectors/bagOfWords/BagOfWords-stopWords.csv")
	X2, _ = getXandY("./vectors/tfIdf/tfidf-stopWords.csv")
	X = np.hstack((X1, X2))
	get_accuracy(X, Y, header + "-stop words removal")

	X1, Y = getXandY("./vectors/bagOfWords/BagOfWords-Lemma.csv")
	X2, _ = getXandY("./vectors/tfIdf/tfidf-Lemma.csv")
	X = np.hstack((X1, X2))
	get_accuracy(X, Y, header + "-Lemmatization")

	X1, Y = getXandY("./vectors/bagOfWords/BagOfWords-stopWords-Lemma.csv")
	X2, _ = getXandY("./vectors/tfIdf/tfidf-stopWords-Lemma.csv")
	X = np.hstack((X1, X2))
	get_accuracy(X, Y, header + "-Lemmatization + stop words removal")

def doc2vecExperiment(useTfidf = False, useBagOfWords = False, useBagOfPOS=False, usePOSWithTfidf=False, stopWordsRemoval=True, useNb=True):
	#no preprocess
	header = "doc2vec"
	if useTfidf:
		header+= "+tfidf"
	if useBagOfWords:
		header+="+bagOfWords"
	if useBagOfPOS:
		header+="+bagOfPos"
	if usePOSWithTfidf:
		header+="+POSWithTfidf"
	X,Y = getXandY(filePath="./vectors/doc2vec/doc2Vec.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
	get_accuracy(X,Y, header+ "-min preprocess", useNb=useNb)

	if stopWordsRemoval:
		#with stop words removal
		X, Y = getXandY(filePath="./vectors/doc2vec/doc2Vec-stopWords.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
		get_accuracy(X, Y, header+ "-stop words removal", useNb=useNb)

	#with Lemmatization
	X,Y =  getXandY(filePath="./vectors/doc2vec/doc2Vec-Lemma.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
	get_accuracy(X, Y, header+ "-Lemmatization", useNb=useNb)

	if stopWordsRemoval:
		#with stop words removal and Lemmatization
		X, Y = getXandY(filePath="./vectors/doc2vec/doc2Vec-stopWords-Lemma.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
		get_accuracy(X, Y, header+"-Lemmatization + stop words removal", useNb=useNb)

def fastTextExperiment(useTfidf = False, useBagOfWords=False, useBagOfPOS=False, usePOSWithTfidf=False, stopWordsRemoval=True, useNb=True):
	#no preprocess
	header = "fastText"
	if useTfidf:
		header+= "+tfidf"
	if useBagOfWords:
		header+="+bagOfWords"
	if useBagOfPOS:
		header+="+bagOfPos"
	if usePOSWithTfidf:
		header+="+POSWithTfidf"

	X,Y = getXandY(filePath="./vectors/fastText/fastText.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
	get_accuracy(X,Y, header+ "-min preprocess", useNb=useNb)
	if stopWordsRemoval:
		#with stop words removal
		X, Y = getXandY(filePath="./vectors/fastText/fastText-stopWords.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
		get_accuracy(X, Y, header+ "-stop words removal", useNb=useNb)

	#with Lemmatization
	X,Y =  getXandY(filePath="./vectors/fastText/fastText-Lemma.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
	get_accuracy(X, Y, header + "-Lemmatization", useNb=useNb)

	if stopWordsRemoval:
		#with stop words removal and Lemmatization
		X, Y = getXandY(filePath="./vectors/fastText/fastText-stopWords-Lemma.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
		get_accuracy(X, Y, header+"-Lemmatization + stop words removal", useNb=useNb)

def inferSentExperiment(useTfidf = False, useBagOfWords=False,  useBagOfPOS=False, usePOSWithTfidf=False, stopWordsRemoval=True, useNb=True):
	#no preprocess
	header = "inferSent"
	if useTfidf:
		header+= "+tfidf"
	if useBagOfWords:
		header+="+bagOfWords"
	if useBagOfPOS:
		header+="+bagOfPos"
	if usePOSWithTfidf:
		header+="+POSWithTfidf"

	X,Y = getXandY(filePath="./vectors/inferSent/inferSent.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
	get_accuracy(X,Y, header+ "-min preprocess", useNb=useNb)

	if stopWordsRemoval:
		#with stop words removal
		X, Y = getXandY(filePath="./vectors/inferSent/inferSent-stopWords.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
		get_accuracy(X, Y, header+ "-stop words removal", useNb=useNb)

	#with Lemmatization
	X,Y =  getXandY(filePath="./vectors/inferSent/inferSent-Lemma.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
	get_accuracy(X, Y, header + "-Lemmatization", useNb=useNb)

	if stopWordsRemoval:
		#with stop words removal and Lemmatization
		X, Y = getXandY(filePath="./vectors/inferSent/inferSent-stopWords-Lemma.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
		get_accuracy(X, Y, header+"-Lemmatization + stop words removal", useNb=useNb)

def miniLMExperiment(useTfidf = False, useBagOfWords=False, useBagOfPOS=False, usePOSWithTfidf=False, stopWordsRemoval=True, useNb=True):
	#no preprocess
	header = "miniLM"
	if useTfidf:
		header+= "+tfidf"
	if useBagOfWords:
		header += "+bagOfWords"
	if useBagOfPOS:
		header+="+bagOfPos"
	if usePOSWithTfidf:
		header+="+POSWithTfidf"

	X,Y = getXandY(filePath="./vectors/MiniLM/miniLM.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
	get_accuracy(X,Y, header+ "-min preprocess", useNb=useNb)

	if stopWordsRemoval:
		#with stop words removal
		X, Y = getXandY(filePath="./vectors/MiniLM/miniLM-stopWords.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
		get_accuracy(X, Y, header+ "-stop words removal", useNb=useNb)

	#with Lemmatization
	X,Y =  getXandY(filePath="./vectors/MiniLM/miniLM-Lemma.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
	get_accuracy(X, Y, header + "-Lemmatization", useNb=useNb)

	if stopWordsRemoval:
		#with stop words removal and Lemmatization
		X, Y = getXandY(filePath="./vectors/MiniLM/miniLM-stopWords-Lemma.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
		get_accuracy(X, Y, header+"-Lemmatization + stop words removal", useNb=useNb)

def mpnetExperiment(useTfidf = False, useBagOfWords=False, useBagOfPOS=False, usePOSWithTfidf=False, stopWordsRemoval=True, useNb=True):
	#no preprocess
	header = "mpnet"
	if useTfidf:
		header+= "+tfidf"
	if useBagOfWords:
		header+="+bagOfWords"
	if useBagOfPOS:
		header+="+bagOfPos"
	if usePOSWithTfidf:
		header+="+POSWithTfidf"

	X,Y = getXandY(filePath="./vectors/mpnet/mpnet.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
	get_accuracy(X,Y, header+ "-min preprocess", useNb=useNb)

	if stopWordsRemoval:
		#with stop words removal
		X, Y = getXandY(filePath="./vectors/mpnet/mpnet-stopWords.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
		get_accuracy(X, Y, header+ "-stop words removal", useNb=useNb)

	#with Lemmatization
	X,Y =  getXandY(filePath="./vectors/mpnet/mpnet-Lemma.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
	get_accuracy(X, Y, header + "-Lemmatization", useNb=useNb)

	if stopWordsRemoval:
		#with stop words removal and Lemmatization
		X, Y = getXandY(filePath="./vectors/mpnet/mpnet-stopWords-Lemma.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
		get_accuracy(X, Y, header+"-Lemmatization + stop words removal", useNb=useNb)

def robertaExperiment(useTfidf = False, useBagOfWords = False, useBagOfPOS=False, usePOSWithTfidf=False, stopWordsRemoval=True, useNb=True):
	#no preprocess
	header = "roberta"
	if useTfidf:
		header+= "+tfidf"
	if useBagOfWords:
		header += "+bagOfWords"
	if useBagOfPOS:
		header+="+bagOfPos"
	if usePOSWithTfidf:
		header+="+POSWithTfidf"

	X,Y = getXandY(filePath="./vectors/Roberta/roberta.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
	get_accuracy(X,Y, header+ "-min preprocess", useNb=useNb)

	if stopWordsRemoval:
		#with stop words removal
		X, Y = getXandY(filePath="./vectors/Roberta/roberta-stopWords.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
		get_accuracy(X, Y, header+ "-stop words removal", useNb=useNb)

	#with Lemmatization
	X,Y =  getXandY(filePath="./vectors/Roberta/roberta-Lemma.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
	get_accuracy(X, Y, header + "-Lemmatization", useNb=useNb)

	if stopWordsRemoval:
		#with stop words removal and Lemmatization
		X, Y = getXandY(filePath="./vectors/Roberta/roberta-stopWords-Lemma.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
		get_accuracy(X, Y, header+"-Lemmatization + stop words removal", useNb=useNb)

def USEExperiment(useTfidf = False, useBagOfWords=False, useBagOfPOS=False, usePOSWithTfidf=False, stopWordsRemoval=True, useNb=True):
	#no preprocess
	header = "USE"
	if useTfidf:
		header+= "+tfidf"
	if useBagOfWords:
		header += "+bagOfWords"
	if useBagOfPOS:
		header+="+bagOfPos"
	if usePOSWithTfidf:
		header+="+POSWithTfidf"

	X,Y = getXandY(filePath="./vectors/USE/USE.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
	get_accuracy(X,Y, header+ "-min preprocess", useNb=useNb)

	if stopWordsRemoval:
		#with stop words removal
		X, Y = getXandY(filePath="./vectors/USE/USE-stopWords.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
		get_accuracy(X, Y, header+ "-stop words removal", useNb=useNb)

	#with Lemmatization
	X,Y =  getXandY(filePath="./vectors/USE/USE-Lemma.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
	get_accuracy(X, Y, header + "-Lemmatization", useNb=useNb)

	if stopWordsRemoval:
		#with stop words removal and Lemmatization
		X, Y = getXandY(filePath="./vectors/USE/USE-stopWords-Lemma.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
		get_accuracy(X, Y, header+"-Lemmatization + stop words removal", useNb=useNb)

def partOfSpeechExperiment():
	header = "bagOfPOS"
	X, Y = getXandY(filePath="./vectors/bagOfPOS/BagOfPOS.csv")
	get_accuracy(X,Y, header+ "-min preprocess", useNb=False)
	X, Y = getXandY(filePath="./vectors/bagOfPOS/BagOfPOS-stopWords.csv")
	get_accuracy(X, Y, header + "-stop words removal", useNb=False)
	X, Y = getXandY(filePath="./vectors/bagOfPOS/BagOfPOS-punct.csv")
	get_accuracy(X, Y, header + "-punct removal", useNb=False)
	X, Y = getXandY(filePath="./vectors/bagOfPOS/BagOfPOS-stopWords-punct.csv")
	get_accuracy(X, Y, header + "-punct + stop words removal", useNb=False)

	header = "POSWithTfIdf"
	X, Y = getXandY(filePath="./vectors/POSWithTfidf/POSWithTfidf.csv")
	get_accuracy(X, Y, header + "-min preprocess", useNb=False)
	X, Y = getXandY(filePath="./vectors/POSWithTfidf/POSWithTfidf-stopWords.csv")
	get_accuracy(X, Y, header + "-stop words removal", useNb=False)
	X, Y = getXandY(filePath="./vectors/POSWithTfidf/POSWithTfidf-punct.csv")
	get_accuracy(X, Y, header + "-punct removal", useNb=False)
	X, Y = getXandY(filePath="./vectors/POSWithTfidf/POSWithTfidf-stopWords-punct.csv")
	get_accuracy(X, Y, header + "-punct + stop words removal", useNb=False)

def runThirdExperiment():
	bagOfWordsExperiment(stopWordsRemoval=False, useNb=False)
	bagOfWordsExperiment(useBagOfPOS=True, stopWordsRemoval=False, useNb=False)
	bagOfWordsExperiment(usePOSWithTfidf=True, stopWordsRemoval=False, useNb=False)
	tfidfExperiment(stopWordsRemoval=False, useNb=False)
	tfidfExperiment(useBagOfPOS=True, stopWordsRemoval=False, useNb=False)
	tfidfExperiment(usePOSWithTfidf=True, stopWordsRemoval=False, useNb=False)
	doc2vecExperiment(stopWordsRemoval=False, useNb=False)
	doc2vecExperiment(useBagOfPOS=True, stopWordsRemoval=False, useNb=False)
	doc2vecExperiment(usePOSWithTfidf=True, stopWordsRemoval=False, useNb=False)
	doc2vecExperiment(useTfidf=True, stopWordsRemoval=False, useNb=False)
	doc2vecExperiment(useTfidf=True, useBagOfPOS=True, stopWordsRemoval=False, useNb=False)
	doc2vecExperiment(useTfidf=True, usePOSWithTfidf=True, stopWordsRemoval=False, useNb=False)
	inferSentExperiment(stopWordsRemoval=False, useNb=False)
	inferSentExperiment(useBagOfPOS=True, stopWordsRemoval=False, useNb=False)
	inferSentExperiment(usePOSWithTfidf=True, stopWordsRemoval=False, useNb=False)
	inferSentExperiment(useTfidf=True, stopWordsRemoval=False, useNb=False)
	inferSentExperiment(useTfidf=True, useBagOfPOS=True, stopWordsRemoval=False, useNb=False)
	inferSentExperiment(useTfidf=True, usePOSWithTfidf=True, stopWordsRemoval=False, useNb=False)
	miniLMExperiment(stopWordsRemoval=False, useNb=False)
	miniLMExperiment(useBagOfPOS=True, stopWordsRemoval=False, useNb=False)
	miniLMExperiment(usePOSWithTfidf=True, stopWordsRemoval=False, useNb=False)
	miniLMExperiment(useTfidf=True, stopWordsRemoval=False, useNb=False)
	miniLMExperiment(useTfidf=True, useBagOfPOS=True, stopWordsRemoval=False, useNb=False)
	miniLMExperiment(useTfidf=True, usePOSWithTfidf=True, stopWordsRemoval=False, useNb=False)
	mpnetExperiment(stopWordsRemoval=False, useNb=False)
	mpnetExperiment(useBagOfPOS=True, stopWordsRemoval=False, useNb=False)
	mpnetExperiment(usePOSWithTfidf=True, stopWordsRemoval=False, useNb=False)
	mpnetExperiment(useTfidf=True, stopWordsRemoval=False, useNb=False)
	mpnetExperiment(useTfidf=True, useBagOfPOS=True, stopWordsRemoval=False, useNb=False)
	mpnetExperiment(useTfidf=True, usePOSWithTfidf=True, stopWordsRemoval=False, useNb=False)
	robertaExperiment(stopWordsRemoval=False, useNb=False)
	robertaExperiment(useBagOfPOS=True, stopWordsRemoval=False, useNb=False)
	robertaExperiment(usePOSWithTfidf=True, stopWordsRemoval=False, useNb=False)
	robertaExperiment(useTfidf=True, stopWordsRemoval=False, useNb=False)
	robertaExperiment(useTfidf=True, useBagOfPOS=True, stopWordsRemoval=False, useNb=False)
	robertaExperiment(useTfidf=True, usePOSWithTfidf=True, stopWordsRemoval=False, useNb=False)
	USEExperiment(stopWordsRemoval=False, useNb=False)
	USEExperiment(useBagOfPOS=True, stopWordsRemoval=False, useNb=False)
	USEExperiment(usePOSWithTfidf=True, stopWordsRemoval=False, useNb=False)
	USEExperiment(useTfidf=True, stopWordsRemoval=False, useNb=False)
	USEExperiment(useTfidf=True, useBagOfPOS=True, stopWordsRemoval=False, useNb=False)
	USEExperiment(useTfidf=True, usePOSWithTfidf=True, stopWordsRemoval=False, useNb=False)

def runInitialExperiment():
	bagOfWordsExperiment()
	tfidfExperiment()
	fastTextExperiment()
	doc2vecExperiment()
	inferSentExperiment()
	miniLMExperiment()
	mpnetExperiment()
	robertaExperiment()
	USEExperiment()

def runSecondExperiment():
	bagOfWordsExperiment()
	tfidfExperiment()
	BagOfWordsWithTfIdfExperiement()
	doc2vecExperiment(stopWordsRemoval=False, useNb=False)
	doc2vecExperiment(useTfidf=False, useBagOfWords=True, stopWordsRemoval=False, useNb=False)
	doc2vecExperiment(useTfidf=True, stopWordsRemoval=False, useNb=False)
	doc2vecExperiment(useTfidf=True, useBagOfWords=True, stopWordsRemoval=False, useNb=False)
	inferSentExperiment(stopWordsRemoval=False, useNb=False)
	inferSentExperiment(useTfidf=False, useBagOfWords=True, stopWordsRemoval=False, useNb=False)
	inferSentExperiment(useTfidf=True, stopWordsRemoval=False, useNb=False)
	inferSentExperiment(useTfidf=True, useBagOfWords=True, stopWordsRemoval=False, useNb=False)
	miniLMExperiment( stopWordsRemoval=False, useNb=False)
	miniLMExperiment(useTfidf=False, useBagOfWords=True, stopWordsRemoval=False, useNb=False)
	miniLMExperiment(useTfidf=True, stopWordsRemoval=False, useNb=False)
	miniLMExperiment(useTfidf=True, useBagOfWords=True, stopWordsRemoval=False, useNb=False)
	mpnetExperiment( stopWordsRemoval=False, useNb=False)
	mpnetExperiment(useTfidf=False, useBagOfWords=True, stopWordsRemoval=False, useNb=False)
	mpnetExperiment(useTfidf=True, stopWordsRemoval=False, useNb=False)
	mpnetExperiment(useTfidf=True, useBagOfWords=True, stopWordsRemoval=False, useNb=False)
	robertaExperiment( stopWordsRemoval=False, useNb=False)
	robertaExperiment(useTfidf=False, useBagOfWords=True,  stopWordsRemoval=False, useNb=False)
	robertaExperiment(useTfidf=True,  stopWordsRemoval=False, useNb=False)
	robertaExperiment(useTfidf=True, useBagOfWords=True,  stopWordsRemoval=False, useNb=False)
	USEExperiment(stopWordsRemoval=False, useNb=False)
	USEExperiment(useTfidf=False, useBagOfWords=True, stopWordsRemoval=False, useNb=False)
	USEExperiment(useTfidf=True,  stopWordsRemoval=False, useNb=False)
	USEExperiment(useTfidf=True, useBagOfWords=True,  stopWordsRemoval=False, useNb=False)


def evluateResults(filePath, checkPreProcess, checkDataRepresent, checkLearningAlgo, checkVectorsAddition):
	if checkPreProcess:
		df = pd.read_csv(filePath)
		df_no_preprocess = df.loc[df["Title"].str.contains("-min preprocess")]
		calcaulateResultsMean(df_no_preprocess, "./results-mean.csv", "min preprocess")
		df_stop_words = df.loc[df["Title"].str.contains("-stop words removal")]
		calcaulateResultsMean(df_stop_words, "./results-mean.csv", "stop words removal")
		df_lemma = df.loc[(df["Title"].str.contains("-Lemmatization") ) & (~df["Title"].str.contains("stop words removal"))]
		calcaulateResultsMean(df_lemma, "./results-mean.csv", "Lemmatization")
		df_lemma_stop_words = df.loc[(df["Title"].str.contains("-Lemmatization") ) & (df["Title"].str.contains("stop words removal"))]
		calcaulateResultsMean(df_lemma_stop_words, "./results-mean.csv", "Lemmatization + stop words removal")
	if checkDataRepresent:
		df = pd.read_csv(filePath)
		df_bagOfWords = df.loc[df["Title"].str.contains("bagOfWords")]
		calcaulateResultsMean(df_bagOfWords, "./results-mean.csv", "bagOfWords")
		print(df_bagOfWords)
		df_tfidf = df.loc[df["Title"].str.contains("tfidf")]
		calcaulateResultsMean(df_tfidf, "./results-mean.csv", "tfidf")
		print(df_tfidf)
		df_fastText = df.loc[df["Title"].str.contains("fastText")]
		calcaulateResultsMean(df_fastText, "./results-mean.csv", "fastText")
		print(df_fastText)
		df_doc2vec = df.loc[df["Title"].str.contains("doc2vec")]
		calcaulateResultsMean(df_doc2vec, "./results-mean.csv", "doc2vec")
		print(df_doc2vec)
		df_inferSent = df.loc[df["Title"].str.contains("inferSent")]
		calcaulateResultsMean(df_inferSent, "./results-mean.csv", "inferSent")
		print(df_inferSent)
		df_miniLM = df.loc[df["Title"].str.contains("miniLM")]
		calcaulateResultsMean(df_miniLM, "./results-mean.csv", "miniLM")
		print(df_miniLM)
		df_mpnet = df.loc[df["Title"].str.contains("mpnet")]
		calcaulateResultsMean(df_mpnet, "./results-mean.csv", "mpnet")
		print(df_mpnet)
		df_roberta = df.loc[df["Title"].str.contains("roberta")]
		calcaulateResultsMean(df_roberta, "./results-mean.csv", "roberta")
		print(df_roberta)
		df_USE = df.loc[df["Title"].str.contains("USE")]
		calcaulateResultsMean(df_USE, "./results-mean.csv", "USE")
		print(df_USE)
	if checkLearningAlgo:
		df = pd.read_csv(filePath)
		calcaulateResultsMean(df, "./results-mean.csv", "LearningAlgo")



	if checkVectorsAddition:
		df = pd.read_csv(filePath)
		df_noAddition = df.loc[(~df["Title"].str.contains("tfidf")) & (~df["Title"].str.contains("bagOfWords"))]
		print(df_noAddition)
		calcaulateResultsMean(df_noAddition, "./results-mean.csv", "no features addition", useNB=False)
		df_bagOfWords = df.loc[(~df["Title"].str.contains("tfidf")) & (df["Title"].str.contains("\+bagOfWords"))]
		print(df_bagOfWords)
		calcaulateResultsMean(df_bagOfWords, "./results-mean.csv", "bagOfWords addition", useNB=False)
		df_tfidf = df.loc[(df["Title"].str.contains("\+tfidf")) & (~df["Title"].str.contains("bagOfWords", case=False))]
		print(df_tfidf)
		calcaulateResultsMean(df_tfidf, "./results-mean.csv", "tfidf addition", useNB=False)
		df_tfidf_bagOfWords = df.loc[(df["Title"].str.contains("tfidf")) & (df["Title"].str.contains("\+bagOfWords"))]
		print(df_tfidf_bagOfWords)
		calcaulateResultsMean(df_tfidf_bagOfWords, "./results-mean.csv", "tfidf+bagOfWordsAddition", useNB=False)


def calcaulateResultsMean(resultsDataFrame, fileName, title, useRD = True ,useMLP= True, useSVM= True, useNB=True):
	size = 0
	if useRD:
		RD_mean = resultsDataFrame["RD"].mean()
		size+=1
	else:
		RD_mean = 0
	if useMLP:
		MLP_mean = resultsDataFrame["MLP"].mean()
		size+=1
	else:
		MLP_mean = 0
	if useSVM:
		SVM_mean = resultsDataFrame["SVM"].mean()
		size+=1
	else:
		SVM_mean = 0
	if useNB:
		NB_mean = resultsDataFrame["NB"].mean()
		size+=1
	else:
		NB_mean = 0
	total_mean= (RD_mean+MLP_mean+SVM_mean+NB_mean)/size
	field_names = ['Title', 'RD_mean', 'MLP_mean', 'SVM_mean', 'NB_mean', 'Total_mean']
	dict = {'Title': title, 'RD_mean': RD_mean, 'MLP_mean': MLP_mean, 'SVM_mean': SVM_mean, 'NB_mean': NB_mean, 'Total_mean': total_mean}
	fileExist = os.path.isfile(fileName)
	with open(fileName, 'a') as f_object:
		dictwriter_object = DictWriter(f_object, fieldnames=field_names)
		if not fileExist:
			dictwriter_object.writeheader()
		# Pass the dictionary as an argument to the Writerow()
		dictwriter_object.writerow(dict)
		# Close the file object
		f_object.close()

def runParamsTuning():
	X, Y = getXandY(filePath="./vectors/USE/USE-Lemma.csv", useTfidf=True)
	best_params.get_best_params_svm(X,Y)
	X, Y = getXandY(filePath="./vectors/USE/USE-Lemma.csv", useTfidf=True, usePOSWithTfidf=True)
	best_params.get_best_params_mlp(X,Y)
	X, Y = getXandY(filePath="./vectors/USE/USE-Lemma.csv", usePOSWithTfidf=True)
	best_params.get_best_params_rf(X,Y)
	X, Y = getXandY(filePath="./vectors/Roberta/roberta.csv", useTfidf=True, usePOSWithTfidf=True)
	best_params.get_best_params_rf(X,Y)

#runInitialExperiment()
evluateResults("./secondExperiment2.csv", checkPreProcess=False, checkDataRepresent=False, checkLearningAlgo=False,checkVectorsAddition=True)
##partOfSpeechExperiment()
#runSecondExperiment()
#runParamsTuning()
#USEExperiment(useTfidf=True, stopWordsRemoval=False, useNb=False)


