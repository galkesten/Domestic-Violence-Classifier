
from LearningAlgorithms import calcAccuracy
import pandas as pd
import numpy as np
import os
from csv import DictWriter

"""
This script run the Experiments we conducted during the project.
The script will ask you which number of Experiment you want to run.
It expects to get one of these numbers as input:
1 / 2 / 3 / 0. 
0 - for running all the experiments.
For each Experiment, A csv  file with all the results will be created under The Results Directory.
The name of the file will be in this format - Experiment{numExperiment}-results-{current date}.csv
Each csv file will be in the following structure:
*The columns names will RD(Random Forest)/MLP/SVM/NB(Naive base).
*There will be also a column with the name 'title' that indicates the row's name. 
*Each row name will mention the preprocessing method and the kind of features that were used.
*Each cell contains the accuracy score for a triple combination- preprocessing method x featuers type x Learning algorithm.
*If the accuracy score for a cell is -1 it means that this combination was not tested during the experiment.

"""


def getXandY(filePath, useTfidf=False, useBagOfWords=False, usePOSWithTfidf = False, useBagOfPOS= False):
	"""
	This func reads feature vectors + classification from a file path and splits the features and the classifications
	to X,Y variables. This function can also concatenate between featuers for the second and the third Experiment.


	:param filePath: A file path to a csv file that contains a vector for each post in the DomesticViolence database.
	It also contains a column with the name 'Label' which contains the classifications for the posts.
	:param useTfidf: if true- the function will concatenate lexical features that was created by TFIDF model.
	:param useBagOfWords:if true- the function will concatenate lexical features that was created by Bag of words model.
	:param usePOSWithTfidf: if true- the function will concatenate lexical features that was created by POSWithTfidf model.
	:param useBagOfPOS: if true- the function will concatenate lexical features that was created by TFIDF model.
	:return: X- A numpy array with 2 dimensions. Each row represents a vector represention of one of the posts.
			Y- A numpy array with 1 dimension. Each row represents the classification for the the same row(same index) in X
	"""
	df = pd.read_csv(filePath)
	Y = np.array(df['Label'])
	df = df.drop(columns=['Label'])
	X = df.to_numpy()
	X = np.delete(X, 0, 1)
	if useTfidf:
		tfidf = pd.read_csv("vectors/TFIDF/tfidf.csv")
		tfidf = tfidf.drop(columns=['Label'])
		tfidf = tfidf.to_numpy()
		tfidf = np.delete(tfidf, 0, 1)
		X = np.hstack((X, tfidf))
	if useBagOfWords:
		bagOfWords = pd.read_csv("vectors/BagOfWords/BagOfWords.csv")
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
		bagOfPOS = pd.read_csv("vectors/BagOfPOS/BagOfPOS.csv")
		bagOfPOS = bagOfPOS.drop(columns=['Label'])
		bagOfPOS = bagOfPOS.to_numpy()
		bagOfPOS = np.delete(bagOfPOS, 0, 1)
		X = np.hstack((X, bagOfPOS))
	return X,Y


def bagOfWordsExperiment(useTfidf = False, useBagOfPOS=False, usePOSWithTfidf=False, stopWordsRemoval=True, useNb=True, numExperiment=1):
	"""
	This function calculates the accuracy score for all the models that uses features that was created by Bag of words.
	The function will load the csv files from vectors/BagOfWords- which are different Bag of words  features vectors that are
	going to be tested in the Experiments. Each kind of features was created after applying different preprocessing method.
	The function can be used also for an integrated vector representaion that contain Bag of words(it can concatenate other features
	to the Bag of words features.
	The function will run the getAcurracy function which will run different kind of learning algorithms on the features,
	and accuracy scores will be written to a csv file, based on numExperiment var.

	:param useTfidf: if true- a TFIDF vector will be concatenated to each Bag of words vector.
	:param useBagOfPOS:  if true- BagOfPOS vector will be concatenated to each Bag of words vector.
	:param usePOSWithTfidf: if true - POSWithTfidf vector will be concatenated to each Bag of words vector.
	:param stopWordsRemoval: if true- stop words removal+Bag of words and stop words removal+Lemmatization+ Bag of words featuers will be tested.
	:param useNb: if true- accuracy score will be calculated with naive Bayes classifier
	:param numExperiment: can be 1/2/3.  the Number of experiment which calls to this function.
	:return: None
	"""

	header = "BagOfWords"
	if useTfidf:
		header+= "+tfidf"
	if useBagOfPOS:
		header+="+BagOfPOS"
	if usePOSWithTfidf:
		header+="+POSWithTfidf"

	#Minimal preprocess
	X,Y = getXandY(filePath="vectors/BagOfWords/BagOfWords.csv", useTfidf=useTfidf, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
	calcAccuracy(X, Y, header + "-min preprocess", useNb=useNb, useMultinomialNB=True, numExperiment=numExperiment)

	if stopWordsRemoval:
		#Stop words removal
		X, Y = getXandY(filePath="vectors/BagOfWords/BagOfWords-stopWords.csv", useTfidf=useTfidf, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
		calcAccuracy(X, Y, header + "-stop words removal", useNb=useNb, useMultinomialNB=True, numExperiment=numExperiment)

	#Lemmatization
	X,Y =  getXandY(filePath="vectors/BagOfWords/BagOfWords-Lemma.csv", useTfidf=useTfidf, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
	calcAccuracy(X, Y, header + "-Lemmatization", useNb=useNb, useMultinomialNB=True, numExperiment=numExperiment)

	if stopWordsRemoval:
		#Stop words removal and Lemmatization
		X, Y = getXandY(filePath="vectors/BagOfWords/BagOfWords-stopWords-Lemma.csv", useTfidf=useTfidf, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
		calcAccuracy(X, Y, header + "-Lemmatization + stop words removal", useNb=useNb, useMultinomialNB=True, numExperiment=numExperiment)


def TFIDFExperiment(useBagOfPOS=False, usePOSWithTfidf=False, stopWordsRemoval=True, useNb=True, numExperiment=1):
	"""
	This function calculates the accuracy score for all the models that uses features that was created by TFIDF.
	The function will load the csv files from vectors/TFIDF- which are different tfidf features vectors that are
	going to be tested in the Experiments. Each kind of features was created after applying different preprocessing method.
	The function can be used also for an integrated vector representaion that contain TFIDF(it can concatenate other features
	to the TFIDF features.
	The function will run the getAcurracy function which will run different kind of learning algorithms on the features,
	and accuracy scores will be written to a csv file, based on numExperiment var.

	:param useBagOfPOS:  if true- BagOfPOS vector will be concatenated to each TFIDF vector.
	:param usePOSWithTfidf: if true - POSWithTfidf vector will be concatenated to each TFIDF vector.
	:param stopWordsRemoval: if true- stop words removal+TFIDF and stop words removal+Lemmatization+ TFIDF featuers will be tested.
	:param useNb: if true- accuracy score will be calculated with naive Bayes classifier
	:param numExperiment: can be 1/2/3.  the Number of experiment which calls to this function.
	:return: None
	"""
	# Minimal preprocess
	header = "TFIDF"
	if useBagOfPOS:
		header+="+BagOfPOS"
	if usePOSWithTfidf:
		header+="+POSWithTfidf"

	X,Y = getXandY(filePath="vectors/TFIDF/tfidf.csv", useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
	calcAccuracy(X, Y, header + "-min preprocess", useNb=useNb, useMultinomialNB=True, numExperiment=numExperiment)
	if stopWordsRemoval:
		# Stop words removal
		X, Y = getXandY(filePath="vectors/TFIDF/tfidf-stopWords.csv", useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
		calcAccuracy(X, Y, header + "-stop words removal", useNb=useNb, useMultinomialNB=True, numExperiment=numExperiment)

	# Lemmatization
	X,Y =  getXandY(filePath="vectors/TFIDF/tfidf-Lemma.csv", useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
	calcAccuracy(X, Y, header + "-Lemmatization", useNb=useNb, useMultinomialNB=True, numExperiment=numExperiment)

	if stopWordsRemoval:
		# Stop words removal and Lemmatization
		X, Y = getXandY(filePath="vectors/TFIDF/tfidf-stopWords-Lemma.csv", useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
		calcAccuracy(X, Y, header + "-Lemmatization + stop words removal", useNb=useNb, useMultinomialNB=True, numExperiment=numExperiment)


def bagOfWordsWithTfIdfExperiement(numExperiment=1):
	"""
		This function calculates the accuracy score for using onlu BagOfWords+TFIDF features.
		 Each kind of features was created after applying different preprocessing method.
		The function will run the getAcurracy function which will run different kind of learning algorithms on the features,
		and accuracy scores will be written to a csv file, based on numExperiment var.

		:param numExperiment: can be 1/2/3.  the Number of experiment which calls to this function.
		:return: None
		"""
	header = "BagOfWords+TFIDF"

	# Minimal preprocess
	X1, Y = getXandY("vectors/BagOfWords/BagOfWords.csv")
	X2, _ = getXandY("vectors/TFIDF/tfidf.csv")
	X = np.hstack((X1, X2))
	calcAccuracy(X, Y, header + "-min preprocess", useMultinomialNB=True, numExperiment=numExperiment)

	# Stop words removal
	X1, Y = getXandY("vectors/BagOfWords/BagOfWords-stopWords.csv")
	X2, _ = getXandY("vectors/TFIDF/tfidf-stopWords.csv")
	X = np.hstack((X1, X2))
	calcAccuracy(X, Y, header + "-stop words removal", useMultinomialNB=True, numExperiment=numExperiment)

	# Lemmatization
	X1, Y = getXandY("vectors/BagOfWords/BagOfWords-Lemma.csv")
	X2, _ = getXandY("vectors/TFIDF/tfidf-Lemma.csv")
	X = np.hstack((X1, X2))
	calcAccuracy(X, Y, header + "-Lemmatization", useMultinomialNB=True, numExperiment=numExperiment)

	# Stop words removal+Lemmatization
	X1, Y = getXandY("vectors/BagOfWords/BagOfWords-stopWords-Lemma.csv")
	X2, _ = getXandY("vectors/TFIDF/tfidf-stopWords-Lemma.csv")
	X = np.hstack((X1, X2))
	calcAccuracy(X, Y, header + "-Lemmatization + stop words removal", useMultinomialNB=True, numExperiment=numExperiment)


def doc2vecExperiment(useTfidf = False, useBagOfWords = False, useBagOfPOS=False, usePOSWithTfidf=False, stopWordsRemoval=True, useNb=True, numExperiment=1):
	"""
		This function calculates the accuracy score for all the models that uses features that was created by doc2vec.
		The function will load the csv files from vectors/doc2vec- which are different Bag of words  features vectors that are
		going to be tested in the Experiments. Each kind of features was created after applying different preprocessing method.
		The function can be used also for an integrated vector representaion that contain doc2vec(it can concatenate other features
		to the doc2vec features.
		The function will run the getAcurracy function which will run different kind of learning algorithms on the features,
		and accuracy scores will be written to a csv file, based on numExperiment var.

		:param useTfidf: if true- a TFIDF vector will be concatenated to each doc2vec vector.
		:param useBagOfWords: if true- a Bag of words vector will be concatenated to each doc2vec vector.
		:param useBagOfPOS:  if true- BagOfPOS vector will be concatenated to each doc2vec vector.
		:param usePOSWithTfidf: if true - POSWithTfidf vector will be concatenated to each doc2vec vector.
		:param stopWordsRemoval: if true- stop words removal+Bag of words and stop words removal+Lemmatization+ doc2vec featuers will be tested.
		:param useNb: if true- accuracy score will be calculated with naive Bayes classifier
		:param numExperiment: can be 1/2/3.  the Number of experiment which calls to this function.
		:return: None
		"""

	header = "doc2vec"
	if useTfidf:
		header+= "+TFIDF"
	if useBagOfWords:
		header+="+BagOfWords"
	if useBagOfPOS:
		header+="+BagOfPOS"
	if usePOSWithTfidf:
		header+="+POSWithTfidf"

	# Minimal preprocess
	X,Y = getXandY(filePath="vectors/doc2vec/doc2vec.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
	calcAccuracy(X, Y, header + "-min preprocess", useNb=useNb, numExperiment=numExperiment)

	if stopWordsRemoval:
		# Stop words removal
		X, Y = getXandY(filePath="vectors/doc2vec/doc2vec-stopWords.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
		calcAccuracy(X, Y, header + "-stop words removal", useNb=useNb, numExperiment=numExperiment)

	# Lemmatization
	X,Y =  getXandY(filePath="vectors/doc2vec/doc2vec-Lemma.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
	calcAccuracy(X, Y, header + "-Lemmatization", useNb=useNb, numExperiment=numExperiment)

	if stopWordsRemoval:
		# Stop words removal and Lemmatization
		X, Y = getXandY(filePath="vectors/doc2vec/doc2vec-stopWords-Lemma.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
		calcAccuracy(X, Y, header + "-Lemmatization + stop words removal", useNb=useNb, numExperiment=numExperiment)


def fastTextExperiment(useTfidf = False, useBagOfWords=False, useBagOfPOS=False, usePOSWithTfidf=False, stopWordsRemoval=True, useNb=True, numExperiment=1):
	"""
		This function calculates the accuracy score for all the models that uses features that was created by fastText.
		The function will load the csv files from vectors/fastText- which are different Bag of words  features vectors that are
		going to be tested in the Experiments. Each kind of features was created after applying different preprocessing method.
		The function can be used also for an integrated vector representaion that contain fastText(it can concatenate other features
		to the fastText features.
		The function will run the getAcurracy function which will run different kind of learning algorithms on the features,
		and accuracy scores will be written to a csv file, based on numExperiment var.

		:param useTfidf: if true- a TFIDF vector will be concatenated to each fastText vector.
		:param useBagOfWords: if true- a Bag of words vector will be concatenated to each fastText vector.
		:param useBagOfPOS:  if true- BagOfPOS vector will be concatenated to each fastText vector.
		:param usePOSWithTfidf: if true - POSWithTfidf vector will be concatenated to each fastText vector.
		:param stopWordsRemoval: if true- stop words removal+Bag of words and stop words removal+Lemmatization+ fastText featuers will be tested.
		:param useNb: if true- accuracy score will be calculated with naive Bayes classifier
		:param numExperiment: can be 1/2/3.  the Number of experiment which calls to this function.
		:return: None
	"""

	header = "fastText"
	if useTfidf:
		header+= "+TFIDF"
	if useBagOfWords:
		header+="+BagOfWords"
	if useBagOfPOS:
		header+="+BagOfPOS"
	if usePOSWithTfidf:
		header+="+POSWithTfidf"

	#Minimal preprocess
	X,Y = getXandY(filePath="./vectors/fastText/fastText.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
	calcAccuracy(X, Y, header + "-min preprocess", useNb=useNb, numExperiment=numExperiment)
	if stopWordsRemoval:
		#Stop words removal
		X, Y = getXandY(filePath="./vectors/fastText/fastText-stopWords.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
		calcAccuracy(X, Y, header + "-stop words removal", useNb=useNb, numExperiment=numExperiment)

	#Lemmatization
	X,Y =  getXandY(filePath="./vectors/fastText/fastText-Lemma.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
	calcAccuracy(X, Y, header + "-Lemmatization", useNb=useNb, numExperiment=numExperiment)

	if stopWordsRemoval:
		#Stop words removal and Lemmatization
		X, Y = getXandY(filePath="./vectors/fastText/fastText-stopWords-Lemma.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
		calcAccuracy(X, Y, header + "-Lemmatization + stop words removal", useNb=useNb, numExperiment=numExperiment)


def inferSentExperiment(useTfidf = False, useBagOfWords=False,  useBagOfPOS=False, usePOSWithTfidf=False, stopWordsRemoval=True, useNb=True, numExperiment=1):
	"""
		This function calculates the accuracy score for all the models that uses features that was created by inferSent.
		The function will load the csv files from vectors/inferSent- which are different Bag of words  features vectors that are
		going to be tested in the Experiments. Each kind of features was created after applying different preprocessing method.
		The function can be used also for an integrated vector representaion that contain inferSent(it can concatenate other features
		to the inferSent features.
		The function will run the getAcurracy function which will run different kind of learning algorithms on the features,
		and accuracy scores will be written to a csv file, based on numExperiment var.

		:param useTfidf: if true- a TFIDF vector will be concatenated to each inferSent vector.
		:param useBagOfWords: if true- a Bag of words vector will be concatenated to each inferSent vector.
		:param useBagOfPOS:  if true- BagOfPOS vector will be concatenated to each inferSent vector.
		:param usePOSWithTfidf: if true - POSWithTfidf vector will be concatenated to each inferSent vector.
		:param stopWordsRemoval: if true- stop words removal+Bag of words and stop words removal+Lemmatization+ inferSent featuers will be tested.
		:param useNb: if true- accuracy score will be calculated with naive Bayes classifier
		:param numExperiment: can be 1/2/3.  the Number of experiment which calls to this function.
		:return: None
		"""
	# Minimal preprocess
	header = "inferSent"
	if useTfidf:
		header+= "+TFIDF"
	if useBagOfWords:
		header+="+BagOfWords"
	if useBagOfPOS:
		header+="+BagOfPOS"
	if usePOSWithTfidf:
		header+="+POSWithTfidf"

	X,Y = getXandY(filePath="./vectors/inferSent/inferSent.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
	calcAccuracy(X, Y, header + "-min preprocess", useNb=useNb, numExperiment=numExperiment)

	if stopWordsRemoval:
		# Stop words removal
		X, Y = getXandY(filePath="./vectors/inferSent/inferSent-stopWords.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
		calcAccuracy(X, Y, header + "-stop words removal", useNb=useNb, numExperiment=numExperiment)

	# Lemmatization
	X,Y =  getXandY(filePath="./vectors/inferSent/inferSent-Lemma.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
	calcAccuracy(X, Y, header + "-Lemmatization", useNb=useNb, numExperiment=numExperiment)

	if stopWordsRemoval:
		# Stop words removal and Lemmatization
		X, Y = getXandY(filePath="./vectors/inferSent/inferSent-stopWords-Lemma.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
		calcAccuracy(X, Y, header + "-Lemmatization + stop words removal", useNb=useNb, numExperiment=numExperiment)


def miniLMExperiment(useTfidf = False, useBagOfWords=False, useBagOfPOS=False, usePOSWithTfidf=False, stopWordsRemoval=True, useNb=True, numExperiment=1):
	"""
		This function calculates the accuracy score for all the models that uses features that was created by MiniLM.
		The function will load the csv files from vectors/MiniLM- which are different Bag of words  features vectors that are
		going to be tested in the Experiments. Each kind of features was created after applying different preprocessing method.
		The function can be used also for an integrated vector representaion that contain MiniLM(it can concatenate other features
		to the MiniLM features.
		The function will run the getAcurracy function which will run different kind of learning algorithms on the features,
		and accuracy scores will be written to a csv file, based on numExperiment var.

		:param useTfidf: if true- a TFIDF vector will be concatenated to each MiniLM vector.
		:param useBagOfWords: if true- a Bag of words vector will be concatenated to each MiniLM vector.
		:param useBagOfPOS:  if true- BagOfPOS vector will be concatenated to each MiniLM vector.
		:param usePOSWithTfidf: if true - POSWithTfidf vector will be concatenated to each MiniLM vector.
		:param stopWordsRemoval: if true- stop words removal+Bag of words and stop words removal+Lemmatization+ MiniLM featuers will be tested.
		:param useNb: if true- accuracy score will be calculated with naive Bayes classifier
		:param numExperiment: can be 1/2/3.  the Number of experiment which calls to this function.
		:return: None
	"""

	header = "MiniLM"
	if useTfidf:
		header+= "+TFIDF"
	if useBagOfWords:
		header += "+BagOfWords"
	if useBagOfPOS:
		header+="+BagOfPOS"
	if usePOSWithTfidf:
		header+="+POSWithTfidf"

	X,Y = getXandY(filePath="./vectors/MiniLM/miniLM.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
	calcAccuracy(X, Y, header + "-min preprocess", useNb=useNb, numExperiment=numExperiment)

	if stopWordsRemoval:
		#Stop words removal
		X, Y = getXandY(filePath="./vectors/MiniLM/miniLM-stopWords.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
		calcAccuracy(X, Y, header + "-stop words removal", useNb=useNb, numExperiment=numExperiment)

	#Lemmatization
	X,Y =  getXandY(filePath="./vectors/MiniLM/miniLM-Lemma.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
	calcAccuracy(X, Y, header + "-Lemmatization", useNb=useNb, numExperiment=numExperiment)

	if stopWordsRemoval:
		#Stop words removal and Lemmatization
		X, Y = getXandY(filePath="./vectors/MiniLM/miniLM-stopWords-Lemma.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
		calcAccuracy(X, Y, header + "-Lemmatization + stop words removal", useNb=useNb, numExperiment=numExperiment)


def mpnetExperiment(useTfidf = False, useBagOfWords=False, useBagOfPOS=False, usePOSWithTfidf=False, stopWordsRemoval=True, useNb=True, numExperiment=1):
	"""
			This function calculates the accuracy score for all the models that uses features that was created by MPNet.
			The function will load the csv files from vectors/MPNet- which are different Bag of words  features vectors that are
			going to be tested in the Experiments. Each kind of features was created after applying different preprocessing method.
			The function can be used also for an integrated vector representaion that contain MPNet(it can concatenate other features
			to the MPNet features.
			The function will run the getAcurracy function which will run different kind of learning algorithms on the features,
			and accuracy scores will be written to a csv file, based on numExperiment var.

			:param useTfidf: if true- a TFIDF vector will be concatenated to each MPNet vector.
			:param useBagOfWords: if true- a Bag of words vector will be concatenated to each MPNet vector.
			:param useBagOfPOS:  if true- BagOfPOS vector will be concatenated to each MiniLM vector.
			:param usePOSWithTfidf: if true - POSWithTfidf vector will be concatenated to each MPNet vector.
			:param stopWordsRemoval: if true- stop words removal+Bag of words and stop words removal+Lemmatization+ MPNet featuers will be tested.
			:param useNb: if true- accuracy score will be calculated with naive Bayes classifier
			:param numExperiment: can be 1/2/3.  the Number of experiment which calls to this function.
			:return: None
		"""

	header = "MPNet"
	if useTfidf:
		header+= "+TFIDF"
	if useBagOfWords:
		header+="+BagOfWords"
	if useBagOfPOS:
		header+="+BagOfPOS"
	if usePOSWithTfidf:
		header+="+POSWithTfidf"

	#Minimal preprocess
	X,Y = getXandY(filePath="vectors/MPNet/mpnet.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
	calcAccuracy(X, Y, header + "-min preprocess", useNb=useNb, numExperiment=numExperiment)

	if stopWordsRemoval:
		#with stop words removal
		X, Y = getXandY(filePath="vectors/MPNet/mpnet-stopWords.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
		calcAccuracy(X, Y, header + "-stop words removal", useNb=useNb, numExperiment=numExperiment)

	#with Lemmatization
	X,Y =  getXandY(filePath="vectors/MPNet/mpnet-Lemma.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
	calcAccuracy(X, Y, header + "-Lemmatization", useNb=useNb, numExperiment=numExperiment)

	if stopWordsRemoval:
		#with stop words removal and Lemmatization
		X, Y = getXandY(filePath="vectors/MPNet/mpnet-stopWords-Lemma.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
		calcAccuracy(X, Y, header + "-Lemmatization + stop words removal", useNb=useNb, numExperiment=numExperiment)


def robertaExperiment(useTfidf = False, useBagOfWords = False, useBagOfPOS=False, usePOSWithTfidf=False, stopWordsRemoval=True, useNb=True, numExperiment=1):
	"""
		This function calculates the accuracy score for all the models that uses features that was created by DistilRoBERTa.
		The function will load the csv files from vectors/Roberta- which are different Bag of words  features vectors that are
		going to be tested in the Experiments. Each kind of features was created after applying different preprocessing method.
		The function can be used also for an integrated vector representaion that contain MPNet(it can concatenate other features
		to the MPNet features.
		The function will run the getAcurracy function which will run different kind of learning algorithms on the features,
		and accuracy scores will be written to a csv file, based on numExperiment var.

		:param useTfidf: if true- a TFIDF vector will be concatenated to each DistilRoBERTa vector.
		:param useBagOfWords: if true- a Bag of words vector will be concatenated to each DistilRoBERTa vector.
		:param useBagOfPOS:  if true- BagOfPOS vector will be concatenated to each DistilRoBERTavector.
		:param usePOSWithTfidf: if true - POSWithTfidf vector will be concatenated to each DistilRoBERTa vector.
		:param stopWordsRemoval: if true- stop words removal+Bag of words and stop words removal+Lemmatization+ DistilRoBERTa featuers will be tested.
		:param useNb: if true- accuracy score will be calculated with naive Bayes classifier
		:param numExperiment: can be 1/2/3.  the Number of experiment which calls to this function.
		:return: None
	"""
	header = "DistilRoBERTa"
	if useTfidf:
		header+= "+TFIDF"
	if useBagOfWords:
		header += "+BagOfWords"
	if useBagOfPOS:
		header+="+BagOfPOS"
	if usePOSWithTfidf:
		header+="+POSWithTfidf"


	X,Y = getXandY(filePath="./vectors/Roberta/roberta.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
	calcAccuracy(X, Y, header + "-min preprocess", useNb=useNb, numExperiment=numExperiment)

	if stopWordsRemoval:
		#with stop words removal
		X, Y = getXandY(filePath="./vectors/Roberta/roberta-stopWords.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
		calcAccuracy(X, Y, header + "-stop words removal", useNb=useNb, numExperiment=numExperiment)

	#with Lemmatization
	X,Y =  getXandY(filePath="./vectors/Roberta/roberta-Lemma.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
	calcAccuracy(X, Y, header + "-Lemmatization", useNb=useNb, numExperiment=numExperiment)

	if stopWordsRemoval:
		#with stop words removal and Lemmatization
		X, Y = getXandY(filePath="./vectors/Roberta/roberta-stopWords-Lemma.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
		calcAccuracy(X, Y, header + "-Lemmatization + stop words removal", useNb=useNb, numExperiment=numExperiment)


def USEExperiment(useTfidf = False, useBagOfWords=False, useBagOfPOS=False, usePOSWithTfidf=False, stopWordsRemoval=True, useNb=True, numExperiment=1):
	"""
		This function calculates the accuracy score for all the models that uses features that was created by Universal Sentence Encoder(USE).
		The function will load the csv files from vectors/USE- which are different Bag of words  features vectors that are
		going to be tested in the Experiments. Each kind of features was created after applying different preprocessing method.
		The function can be used also for an integrated vector representaion that contain USE(it can concatenate other features
		to the USE features.
		The function will run the getAcurracy function which will run different kind of learning algorithms on the features,
		and accuracy scores will be written to a csv file, based on numExperiment var.

		:param useTfidf: if true- a TFIDF vector will be concatenated to each USE vector.
		:param useBagOfWords: if true- a Bag of words vector will be concatenated to each USE vector.
		:param useBagOfPOS:  if true- BagOfPOS vector will be concatenated to each USE vector.
		:param usePOSWithTfidf: if true - POSWithTfidf vector will be concatenated to each USE vector.
		:param stopWordsRemoval: if true- stop words removal+Bag of words and stop words removal+Lemmatization+ USE featuers will be tested.
		:param useNb: if true- accuracy score will be calculated with naive Bayes classifier
		:param numExperiment: can be 1/2/3.  the Number of experiment which calls to this function.
		:return: None
	"""

	header = "Universal Sentence Encoder"
	if useTfidf:
		header+= "+TFIDF"
	if useBagOfWords:
		header += "+BagOfWords"
	if useBagOfPOS:
		header+="+BagOfPOS"
	if usePOSWithTfidf:
		header+="+POSWithTfidf"

	# Minimal preprocess
	X,Y = getXandY(filePath="./vectors/USE/USE.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
	calcAccuracy(X, Y, header + "-min preprocess", useNb=useNb, numExperiment=numExperiment)

	if stopWordsRemoval:
		# Stop words removal
		X, Y = getXandY(filePath="./vectors/USE/USE-stopWords.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
		calcAccuracy(X, Y, header + "-stop words removal", useNb=useNb, numExperiment=numExperiment)

	# Lemmatization
	X,Y =  getXandY(filePath="./vectors/USE/USE-Lemma.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
	calcAccuracy(X, Y, header + "-Lemmatization", useNb=useNb, numExperiment=numExperiment)

	if stopWordsRemoval:
		# Stop words removal and Lemmatization
		X, Y = getXandY(filePath="./vectors/USE/USE-stopWords-Lemma.csv", useTfidf=useTfidf, useBagOfWords=useBagOfWords, useBagOfPOS=useBagOfPOS, usePOSWithTfidf=usePOSWithTfidf)
		calcAccuracy(X, Y, header + "-Lemmatization + stop words removal", useNb=useNb, numExperiment=numExperiment)


def partOfSpeechExperiment():
	"""
		This function calculates the accuracy score for using onlu syntactic features.
		 Each kind of syntatcic features was created after applying different preprocessing method.
		The function will run the getAcurracy function which will run different kind of learning algorithms on the features,
		and accuracy scores will be written to a csv file, based on numExperiment var.

		:return: None
		"""
	header = "BagOfPOS"
	X, Y = getXandY(filePath="vectors/BagOfPOS/BagOfPOS.csv")
	calcAccuracy(X, Y, header + "-min preprocess", useNb=False, numExperiment=3)
	X, Y = getXandY(filePath="vectors/BagOfPOS/BagOfPOS-stopWords.csv")
	calcAccuracy(X, Y, header + "-stop words removal", useNb=False, numExperiment=3)
	X, Y = getXandY(filePath="vectors/BagOfPOS/BagOfPOS-punct.csv")
	calcAccuracy(X, Y, header + "-punct removal", useNb=False, numExperiment=3)
	X, Y = getXandY(filePath="vectors/BagOfPOS/BagOfPOS-stopWords-punct.csv")
	calcAccuracy(X, Y, header + "-punct + stop words removal", useNb=False, numExperiment=3)

	header = "POSWithTfIdf"
	X, Y = getXandY(filePath="./vectors/POSWithTfidf/POSWithTfidf.csv")
	calcAccuracy(X, Y, header + "-min preprocess", useNb=False, numExperiment=3)
	X, Y = getXandY(filePath="./vectors/POSWithTfidf/POSWithTfidf-stopWords.csv")
	calcAccuracy(X, Y, header + "-stop words removal", useNb=False, numExperiment=3)
	X, Y = getXandY(filePath="./vectors/POSWithTfidf/POSWithTfidf-punct.csv")
	calcAccuracy(X, Y, header + "-punct removal", useNb=False, numExperiment=3)
	X, Y = getXandY(filePath="./vectors/POSWithTfidf/POSWithTfidf-stopWords-punct.csv")
	calcAccuracy(X, Y, header + "-punct + stop words removal", useNb=False, numExperiment=3)


def runFirstExperiment():
	"""
	This function will run Experiment number 1.
	The results of this experiment will be written to a csv file that is saved under Results Directory.
	The name of the file will be in this format - Experiment1-results-{current date}.csv
	The csv file will be in the following structure:
	*The columns names will RD(Random Forest)/MLP/SVM/NB(Naive base).
	*There will be also a column with the name 'title' that indicates the row's name.
	*Each row name will mention the preprocessing method and the kind of features that were used.
	*Each cell contains the accuracy score for a triple combination- preprocessing method x featuers type x Learning algorithm.
	:return:
	"""
	bagOfWordsExperiment(numExperiment=1)
	TFIDFExperiment(numExperiment=1)
	fastTextExperiment(numExperiment=1)
	doc2vecExperiment(numExperiment=1)
	inferSentExperiment(numExperiment=1)
	miniLMExperiment(numExperiment=1)
	mpnetExperiment(numExperiment=1)
	robertaExperiment(numExperiment=1)
	USEExperiment(numExperiment=1)

def runSecondExperiment():
	"""
			This function will run Experiment number 2.
			The results of this experiment will be written to a csv file that is saved under Results Directory.
			The name of the file will be in this format - Experiment2-results-{current date}.csv
			The csv file will be in the following structure:
			*The columns names will RD(Random Forest)/MLP/SVM/NB(Naive base).
			*There will be also a column with the name 'title' that indicates the row's name.
			*Each row name will mention the preprocessing method and the kind of features that were used.
			*Each cell contains the accuracy score for a triple combination- preprocessing method x featuers type x Learning algorithm.
			*If the accuracy score for a cell is -1 it means that this combination was not tested during the experiment(This relevant to
			the naive bayes column.
			:return:
			"""
	bagOfWordsExperiment(numExperiment=2)
	TFIDFExperiment(numExperiment=2)
	bagOfWordsWithTfIdfExperiement(numExperiment=2)
	doc2vecExperiment(stopWordsRemoval=False, useNb=False, numExperiment=2)
	doc2vecExperiment(useTfidf=False, useBagOfWords=True, stopWordsRemoval=False, useNb=False, numExperiment=2)
	doc2vecExperiment(useTfidf=True, stopWordsRemoval=False, useNb=False, numExperiment=2)
	doc2vecExperiment(useTfidf=True, useBagOfWords=True, stopWordsRemoval=False, useNb=False, numExperiment=2)
	inferSentExperiment(stopWordsRemoval=False, useNb=False, numExperiment=2)
	inferSentExperiment(useTfidf=False, useBagOfWords=True, stopWordsRemoval=False, useNb=False, numExperiment=2)
	inferSentExperiment(useTfidf=True, stopWordsRemoval=False, useNb=False, numExperiment=2)
	inferSentExperiment(useTfidf=True, useBagOfWords=True, stopWordsRemoval=False, useNb=False, numExperiment=2)
	miniLMExperiment( stopWordsRemoval=False, useNb=False, numExperiment=2)
	miniLMExperiment(useTfidf=False, useBagOfWords=True, stopWordsRemoval=False, useNb=False, numExperiment=2)
	miniLMExperiment(useTfidf=True, stopWordsRemoval=False, useNb=False, numExperiment=2)
	miniLMExperiment(useTfidf=True, useBagOfWords=True, stopWordsRemoval=False, useNb=False, numExperiment=2)
	mpnetExperiment( stopWordsRemoval=False, useNb=False, numExperiment=2)
	mpnetExperiment(useTfidf=False, useBagOfWords=True, stopWordsRemoval=False, useNb=False, numExperiment=2)
	mpnetExperiment(useTfidf=True, stopWordsRemoval=False, useNb=False, numExperiment=2)
	mpnetExperiment(useTfidf=True, useBagOfWords=True, stopWordsRemoval=False, useNb=False, numExperiment=2)
	robertaExperiment( stopWordsRemoval=False, useNb=False, numExperiment=2)
	robertaExperiment(useTfidf=False, useBagOfWords=True,  stopWordsRemoval=False, useNb=False, numExperiment=2)
	robertaExperiment(useTfidf=True,  stopWordsRemoval=False, useNb=False, numExperiment=2)
	robertaExperiment(useTfidf=True, useBagOfWords=True,  stopWordsRemoval=False, useNb=False, numExperiment=2)
	USEExperiment(stopWordsRemoval=False, useNb=False, numExperiment=2)
	USEExperiment(useTfidf=False, useBagOfWords=True, stopWordsRemoval=False, useNb=False, numExperiment=2)
	USEExperiment(useTfidf=True,  stopWordsRemoval=False, useNb=False, numExperiment=2)
	USEExperiment(useTfidf=True, useBagOfWords=True,  stopWordsRemoval=False, useNb=False, numExperiment=2)


def runThirdExperiment():
	"""
		This function will run Experiment number 3.
		The results of this experiment will be written to a csv file that is saved under Results Directory.
		The name of the file will be in this format - Experiment3-results-{current date}.csv
		The csv file will be in the following structure:
		*The columns names will RD(Random Forest)/MLP/SVM/NB(Naive base).
		*There will be also a column with the name 'title' that indicates the row's name.
		*Each row name will mention the preprocessing method and the kind of features that were used.
		*Each cell contains the accuracy score for a triple combination- preprocessing method x featuers type x Learning algorithm.
		*If the accuracy score for a cell is -1 it means that this combination was not tested during the experiment(This relevant to
		the naive bayes column.
		:return:
		"""
	partOfSpeechExperiment()
	bagOfWordsExperiment(stopWordsRemoval=False, useNb=False, numExperiment=3)
	bagOfWordsExperiment(useBagOfPOS=True, stopWordsRemoval=False, useNb=False, numExperiment=3)
	bagOfWordsExperiment(usePOSWithTfidf=True, stopWordsRemoval=False, useNb=False, numExperiment=3)
	TFIDFExperiment(stopWordsRemoval=False, useNb=False, numExperiment=3)
	TFIDFExperiment(useBagOfPOS=True, stopWordsRemoval=False, useNb=False, numExperiment=3)
	TFIDFExperiment(usePOSWithTfidf=True, stopWordsRemoval=False, useNb=False, numExperiment=3)
	doc2vecExperiment(stopWordsRemoval=False, useNb=False, numExperiment=3)
	doc2vecExperiment(useBagOfPOS=True, stopWordsRemoval=False, useNb=False, numExperiment=3)
	doc2vecExperiment(usePOSWithTfidf=True, stopWordsRemoval=False, useNb=False, numExperiment=3)
	doc2vecExperiment(useTfidf=True, stopWordsRemoval=False, useNb=False, numExperiment=3)
	doc2vecExperiment(useTfidf=True, useBagOfPOS=True, stopWordsRemoval=False, useNb=False, numExperiment=3)
	doc2vecExperiment(useTfidf=True, usePOSWithTfidf=True, stopWordsRemoval=False, useNb=False, numExperiment=3)
	inferSentExperiment(stopWordsRemoval=False, useNb=False, numExperiment=3)
	inferSentExperiment(useBagOfPOS=True, stopWordsRemoval=False, useNb=False, numExperiment=3)
	inferSentExperiment(usePOSWithTfidf=True, stopWordsRemoval=False, useNb=False, numExperiment=3)
	inferSentExperiment(useTfidf=True, stopWordsRemoval=False, useNb=False, numExperiment=3)
	inferSentExperiment(useTfidf=True, useBagOfPOS=True, stopWordsRemoval=False, useNb=False, numExperiment=3)
	inferSentExperiment(useTfidf=True, usePOSWithTfidf=True, stopWordsRemoval=False, useNb=False, numExperiment=3)
	miniLMExperiment(stopWordsRemoval=False, useNb=False, numExperiment=3)
	miniLMExperiment(useBagOfPOS=True, stopWordsRemoval=False, useNb=False, numExperiment=3)
	miniLMExperiment(usePOSWithTfidf=True, stopWordsRemoval=False, useNb=False, numExperiment=3)
	miniLMExperiment(useTfidf=True, stopWordsRemoval=False, useNb=False, numExperiment=3)
	miniLMExperiment(useTfidf=True, useBagOfPOS=True, stopWordsRemoval=False, useNb=False, numExperiment=3)
	miniLMExperiment(useTfidf=True, usePOSWithTfidf=True, stopWordsRemoval=False, useNb=False, numExperiment=3)
	mpnetExperiment(stopWordsRemoval=False, useNb=False, numExperiment=3)
	mpnetExperiment(useBagOfPOS=True, stopWordsRemoval=False, useNb=False, numExperiment=3)
	mpnetExperiment(usePOSWithTfidf=True, stopWordsRemoval=False, useNb=False, numExperiment=3)
	mpnetExperiment(useTfidf=True, stopWordsRemoval=False, useNb=False, numExperiment=3)
	mpnetExperiment(useTfidf=True, useBagOfPOS=True, stopWordsRemoval=False, useNb=False, numExperiment=3)
	mpnetExperiment(useTfidf=True, usePOSWithTfidf=True, stopWordsRemoval=False, useNb=False, numExperiment=3)
	robertaExperiment(stopWordsRemoval=False, useNb=False, numExperiment=3)
	robertaExperiment(useBagOfPOS=True, stopWordsRemoval=False, useNb=False, numExperiment=3)
	robertaExperiment(usePOSWithTfidf=True, stopWordsRemoval=False, useNb=False, numExperiment=3)
	robertaExperiment(useTfidf=True, stopWordsRemoval=False, useNb=False, numExperiment=3)
	robertaExperiment(useTfidf=True, useBagOfPOS=True, stopWordsRemoval=False, useNb=False, numExperiment=3)
	robertaExperiment(useTfidf=True, usePOSWithTfidf=True, stopWordsRemoval=False, useNb=False, numExperiment=3)
	USEExperiment(stopWordsRemoval=False, useNb=False, numExperiment=3)
	USEExperiment(useBagOfPOS=True, stopWordsRemoval=False, useNb=False, numExperiment=3)
	USEExperiment(usePOSWithTfidf=True, stopWordsRemoval=False, useNb=False, numExperiment=3)
	USEExperiment(useTfidf=True, stopWordsRemoval=False, useNb=False, numExperiment=3)
	USEExperiment(useTfidf=True, useBagOfPOS=True, stopWordsRemoval=False, useNb=False, numExperiment=3)
	USEExperiment(useTfidf=True, usePOSWithTfidf=True, stopWordsRemoval=False, useNb=False, numExperiment=3)


numExperiment = input("Please enter the number of experiment you want to run. If you enter 0, all the experiments will be running: ")
if int(numExperiment) == 1:
	runFirstExperiment()
if int(numExperiment) == 2:
	runSecondExperiment()
if int(numExperiment) == 3:
	runThirdExperiment()
if int(numExperiment) == 0:
	runFirstExperiment()
	runSecondExperiment()
	runThirdExperiment()

