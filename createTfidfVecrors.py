from PreProcessor import PreProcessor
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


"""
This scripts creates feature vectors for each post in our DomesticViolence database by using 
TFIDF model. The bag of words model is implemented by sklearn package.
The script will create 4 different Bag of words features due to 4 different pre processing methods.
Minimal Process + TFIDF- vectors will be saved as csv file under this path: vectors/TFIDF/tfidf.csv
Stop words removal + TFIDF- vectors will be saved as csv file under this path: vectors/TFIDF/tfidf-stopWords.csv
Lemmatization + TFIDF- vectors will be saved as csv file under this path: vectors/TFIDF/tfidf-Lemma.csv
Stop words removal + TFIDF + Bag of words -vectors will be saved as csv file under this path: vectors/TFIDF/tfidf-stopWords-Lemma.csv
"""
def createTfidfVectors(removeStopWords, useLemmatization, removePunct=False):
    """"
        This function creates vector from each post in the DomesticViolence database. The vector is created by using
        the TFIDF model. Before creating the vectors, this function calls to the preprocessor class and all the
        posts in the database are being preprocessed.
        After the vectors creation, the vectors will be saved in csv file under vectors/TFIDF.
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
        vectorizer = TfidfVectorizer(stop_words='english')
    else:
        vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(preProcessor.X)
    df = pd.DataFrame(X.toarray())
    df['Label'] = preProcessor.Y
    stop_words_header = "stopWords"
    lemma_header = "Lemma"
    punct_header = "punct"
    file_name = "tfidf"
    if removeStopWords:
        file_name += "-" + stop_words_header
    if useLemmatization:
        file_name += '-' + lemma_header
    if removePunct:
        file_name += '-' + punct_header
    df.to_csv(f'./vectors/tfidf/{file_name}.csv')



# To create Tfidf vectors with Minimal preprocess
createTfidfVectors(removeStopWords=False, useLemmatization=False)


# To create Tfidf vectors after removing stop words
createTfidfVectors(removeStopWords=True, useLemmatization=False)

# To create Tfidf vectors after using Lemmaization
createTfidfVectors(removeStopWords=False, useLemmatization=True)

# To create Tfidf vectors after using Lemmaization and removing stop words
createTfidfVectors(removeStopWords=True, useLemmatization=True)