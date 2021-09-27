from PreProcessor import PreProcessor
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def createBagOfWordsVectors(removeStopWords, useLemmatization, removePunct=False):
	preProcessor = PreProcessor(removeStopWords=False, useLemmatization=useLemmatization, removePunct=removePunct)
	preProcessor.splitDbToXandY()
	if useLemmatization or removePunct:
		preProcessor.cleanPosts()
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
	df.to_csv(f'./vectors/bagOfWords/{file_name}.csv')



# To create BagOfWords vectors with no preprocess - remove the next comment
createBagOfWordsVectors(removeStopWords=False, useLemmatization=False)


#To create BagOfWords vectors after removing stop words - remove the next comment
createBagOfWordsVectors(removeStopWords=True, useLemmatization=False)

#To create BagOfWords vectors after using Lemmaization - remove the next comment
createBagOfWordsVectors(removeStopWords=False, useLemmatization=True)

#To createBagOfWords vectors after using Lemmaization and removing stop words - remove the next comment
createBagOfWordsVectors(removeStopWords=True, useLemmatization=True)