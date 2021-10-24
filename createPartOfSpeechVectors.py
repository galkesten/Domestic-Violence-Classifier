from PreProcessor import PreProcessor
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


"""
This script creates syntactic feature vectors for each post in our DomesticViolence database by using 
BagOfPos and POSWithTFIDF model.
The script will create 4 different kind of features for each model due to 4 different pre processing methods.
For BagOfPOS:
Minimal Process + BagOfPOS- vectors will be saved as csv file under this path: vectors/BagOfPOS/BagOfPOS.csv
Stop words removal + BagOfPOS- vectors will be saved as csv file under this path: vectors/BagOfPOS/BagOfPOS-stopWords.csv
Punctuation removal + BagOfPOS- vectors will be saved as csv file under this path: vectors/BagOfPOS/BagOfPOS-punct.csv
Stop words removal + Punctuation removal +  BagOfPOS -vectors will be saved as csv file under this path: vectors/BagOfPOS/BagOfPOS-stopWords-punct.csv

For POSWithTFIDF:
Minimal Process + POSWithTFIDF- vectors will be saved as csv file under this path: vectors/POSWithTfidf/POSWithTfidf.csv
Stop words removal + POSWithTFIDF- vectors will be saved as csv file under this path: vectors/POSWithTfidf/POSWithTfidf-stopWords.csv
Punctuation removal +POSWithTFIDF- vectors will be saved as csv file under this path: vectors/POSWithTfidf/POSWithTfidf-punct.csv
Stop words removal + Punctuation removal +POSWithTFIDF-vectors will be saved as csv file under this path: vectors/POSWithTfidf/POSWithTfidf-stopWords-punct.csv

"""

def createBagOfPOSVectors(removeStopWords, removePunct=False):
	"""
		This function creates vector from each post in the DomesticViolence database. The vector is created by using
		the BagOfPOS model. Before creating the vectors, this function calls to the preprocessor class and all the
		posts in the database are being preprocessed.
		After the vectors creation, the vectors will be saved in csv file under vectors/BagOfPOS.
		The file name will include the preprocessing method that was used.

		:param removeStopWords: boolean variable that indicates whether or not preform stop words removal
		:param useLemmatization: boolean variable that indicates whether or not preform Lemmatization of each token in each post
		:param removePunct: boolean variable that indicates whether or not preform punctuation removal from each post
		:return: None

	"""
	nlp = spacy.load('en_core_web_sm')
	preProcessor = PreProcessor(removeStopWords=removeStopWords, useLemmatization=False, removePunct=removePunct)
	preProcessor.splitDbToXandY()
	preProcessor.getPreProcessedPostsAsStrings()
	X = preProcessor.X
	docs_list = []
	docs_list = []
	for post in X:
		doc = nlp(post)
		pos_array = doc.to_array("POS")
		templist = [str(item) for item in pos_array]
		docs_list.append(" ".join(templist))
	vectorizer = CountVectorizer()
	X = vectorizer.fit_transform(docs_list)
	df = pd.DataFrame(X.toarray())
	df['Label'] = preProcessor.Y
	stop_words_header = "stopWords"
	punct_header = "punct"
	file_name = "BagOfPOS"
	if removeStopWords:
		file_name += "-" + stop_words_header
	if removePunct:
		file_name += '-' + punct_header
	df.to_csv(f'./vectors/BagOfPOS/{file_name}.csv')

def createPOSWithTfidfVectors(removeStopWords, removePunct=False):
	"""
		This function creates vector from each post in the DomesticViolence database. The vector is created by using
		the POSWithTFIDF model. Before creating the vectors, this function calls to the preprocessor class and all the
		posts in the database are being preprocessed.
		After the vectors creation, the vectors will be saved in csv file under vectors/POSWithTfidf.
		The file name will include the preprocessing method that was used.

		:param removeStopWords: boolean variable that indicates whether or not preform stop words removal
		:param useLemmatization: boolean variable that indicates whether or not preform Lemmatization of each token in each post
		:param removePunct: boolean variable that indicates whether or not preform punctuation removal from each post
		:return: None

		"""
	nlp = spacy.load('en_core_web_sm')
	preProcessor = PreProcessor(removeStopWords=removeStopWords, useLemmatization=False, removePunct=removePunct)
	preProcessor.splitDbToXandY()
	preProcessor.getPreProcessedPostsAsStrings()
	X = preProcessor.X
	docs_list = []
	for post in X:
		doc = nlp(post)
		pos_array = doc.to_array("POS")
		templist = [str(item) for item in pos_array]
		docs_list.append(" ".join(templist))
	vectorizer = TfidfVectorizer()
	X = vectorizer.fit_transform(docs_list)
	df = pd.DataFrame(X.toarray())
	df['Label'] = preProcessor.Y
	stop_words_header = "stopWords"
	punct_header = "punct"
	file_name = "POSWithTfidf"
	if removeStopWords:
		file_name += "-" + stop_words_header
	if removePunct:
		file_name += '-' + punct_header
	df.to_csv(f'./vectors/POSWithTfidf/{file_name}.csv')




# To create BagOfPOS vectors with Minimal preprocess
createBagOfPOSVectors(removeStopWords=False, removePunct=False)


#To create BagOfPOS vectors after removing stop words
createBagOfPOSVectors(removeStopWords=True,  removePunct=False)

#To create BagOfPOS vectors after removing punct
createBagOfPOSVectors(removeStopWords=False, removePunct=True)

#To create BagOfPOS vectors after removing punct and stop words
createBagOfPOSVectors(removeStopWords=True, removePunct=True)


# To create POSWithTfidf vectors with Minimal  preprocess
createPOSWithTfidfVectors(removeStopWords=False,  removePunct=False)

# To create POSWithTfidf vectors after removing stop words
createPOSWithTfidfVectors(removeStopWords=True,  removePunct=False)

# To create POSWithTfidf vectors after after removing punct
createPOSWithTfidfVectors(removeStopWords=False,  removePunct=True)

# To create POSWithTfidf vectors after after removing punct and stop words
createPOSWithTfidfVectors(removeStopWords=True, removePunct=True)