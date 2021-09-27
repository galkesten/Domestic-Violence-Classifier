from PreProcessor import PreProcessor
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def createBagOfPOSVectors(removeStopWords, removePunct=False):
	nlp = spacy.load('en_core_web_sm')
	preProcessor = PreProcessor(removeStopWords=removeStopWords, useLemmatization=False, removePunct=removePunct)
	preProcessor.splitDbToXandY()
	preProcessor.cleanPosts()
	X = preProcessor.X
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
	df.to_csv(f'./vectors/bagOfPOS/{file_name}.csv')

def createPOSWithTfidfVectors(removeStopWords, removePunct=False):
	nlp = spacy.load('en_core_web_sm')
	preProcessor = PreProcessor(removeStopWords=removeStopWords, useLemmatization=False, removePunct=removePunct)
	preProcessor.splitDbToXandY()
	preProcessor.cleanPosts()
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




# To create BagOfPOS vectors with no preprocess - remove the next comment
createBagOfPOSVectors(removeStopWords=False, removePunct=False)


#To create BagOfPOS vectors after removing stop words - remove the next comment
createBagOfPOSVectors(removeStopWords=True,  removePunct=False)

#To create BagOfPOS vectors after removing punct - remove the next comment
createBagOfPOSVectors(removeStopWords=False, removePunct=True)

#To create BagOfPOS vectors after removing punct and stop words - remove the next comment
createBagOfPOSVectors(removeStopWords=True, removePunct=True)


# To create POSWithTfidf vectors with no preprocess - remove the next comment
createPOSWithTfidfVectors(removeStopWords=False,  removePunct=False)

#To create POSWithTfidf vectors after removing stop words - remove the next comment
createPOSWithTfidfVectors(removeStopWords=True,  removePunct=False)

#To create POSWithTfidf vectors after after removing punct - remove the next comment
createPOSWithTfidfVectors(removeStopWords=False,  removePunct=True)

#To create POSWithTfidf vectors after after removing punct and stop words - remove the next comment
createPOSWithTfidfVectors(removeStopWords=True, removePunct=True)