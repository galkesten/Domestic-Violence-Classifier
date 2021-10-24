from PreProcessor import PreProcessor
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

"""
This script creates feature vectors for each post in our DomesticViolence database by using 
Bag of words model. The bag of words model is implemented by sklearn package.
The script will create 4 different Bag of words features due to 4 different pre processing methods.
Minimal Process + Bag of words- vectors will be saved as csv file under this path: ./vectors/BagOfWords/BagOfWords.csv
Stop words removal + Bag of words- vectors will be saved as csv file under this path: ./vectors/BagOfWords/BagOfWords-stopWords.csv
Lemmatization + Bag of words- vectors will be saved as csv file under this path: ./vectors/BagOfWords/BagOfWords-Lemma.csv
Stop words removal + Lemmatization + Bag of words -vectors will be saved as csv file under this path: ./vectors/BagOfWords/BagOfWords-stopWords-Lemma.csv
"""


def createBagOfWordsVectors(removeStopWords, useLemmatization, removePunct=False):
	"""
		This function creates vector from each post in the DomesticViolence database. The vector is created by using
		the bag of words model. Before creating the vectors, this function calls to the preprocessor class and all the
		posts in the database are being preprocessed.
		After the vectors creation, the vectors will be saved in csv file under vectors/BagOfWords.
		The file name will include the preprocessing method that was used.

		:param removeStopWords: boolean variable that indicates whether or not preform stop words removal
		:param useLemmatization: boolean variable that indicates whether or not preform Lemmatization of each token in each post
		:param removePunct: boolean variable that indicates whether or not preform punctuation removal from each post
		:return: None

"""
	preProcessor = PreProcessor(removeStopWords=False, useLemmatization=useLemmatization, removePunct=removePunct)
	preProcessor.splitDbToXandY()
	if useLemmatization or removePunct:
		preProcessor.getPreProcessedPostsAsStrings()
	if removeStopWords:
		vectorizer = CountVectorizer(stop_words='english')
	else:
		vectorizer = CountVectorizer()
	X = vectorizer.fit_transform(preProcessor.X)
	df = pd.DataFrame(X.toarray())
	df['Label'] = preProcessor.Y
	stop_words_header = "stopWords"
	lemma_header = "Lemma"
	punct_header = "punct"
	file_name = "BagOfWords"
	if removeStopWords:
		file_name += "-" + stop_words_header
	if useLemmatization:
		file_name += '-' + lemma_header
	if removePunct:
		file_name += '-' + punct_header
	df.to_csv(f'./vectors/BagOfWords/{file_name}.csv')


# To create BagOfWords vectors with Minimal preprocess
createBagOfWordsVectors(removeStopWords=False, useLemmatization=False)

# To create BagOfWords vectors after removing stop words
createBagOfWordsVectors(removeStopWords=True, useLemmatization=False)

# To create BagOfWords vectors after using Lemmatization - remove the next comment
createBagOfWordsVectors(removeStopWords=False, useLemmatization=True)

# To createBagOfWords vectors after using Lemmatization and removing stop words
createBagOfWordsVectors(removeStopWords=True, useLemmatization=True)
