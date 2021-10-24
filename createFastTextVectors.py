from PreProcessor import PreProcessor
from gensim.models.fasttext import load_facebook_model
import numpy as np
import pandas as pd

"""
This script creates feature vectors for each post in our DomesticViolence database by using 
fastText model. The fast text model is pre trained and in order to run you need to download the file cc.en.300.bin.gz
from https://fasttext.cc/docs/en/crawl-vectors.html and save it under vectors/fastText/cc.en.300.bin.gz.
We used genism package to load fastText model.
The script will create 4 different fast Text features due to 4 different pre processing methods.
Minimal Process + fast text- vectors will be saved as csv file under this path: vectors/fastText/fastText.csv
Stop words removal + Bag of words- vectors will be saved as csv file under this path: vectors/fastText/fastText-stopWords.csv
Lemmatization + Bag of words- vectors will be saved as csv file under this path: vectors/fastText/fastText-Lemma.csv
Stop words removal + Lemmatization + Bag of words -vectors will be saved as csv file under this path: vectors/fastText/fastText-stopWords-Lemma.csv
"""

def createFastTextVectors(removeStopWords, useLemmatization, removePunct=False):
    """

    This function creates vector from each post in the DomesticViolence database. The vector is created by using
    the fast text model. Before creating the vectors, this function calls to the preprocessor class and all the
    posts in the database are being preprocessed.
    After the vectors creation, the vectors will be saved in csv file under vectors/fastText.
    The file name will include the preprocessing method that was used.

    :param removeStopWords: boolean variable that indicates whether or not preform stop words removal
    :param useLemmatization: boolean variable that indicates whether or not preform Lemmatization of each token in each post
    :param removePunct: boolean variable that indicates whether or not preform punctuation removal from each post
    :return: None

    """

    preProcessor = PreProcessor(removeStopWords=removeStopWords, useLemmatization=useLemmatization, removePunct=removePunct)
    preProcessor.splitDbToXandY()
    preProcessor.getPreProcessedPostsAsTokens()
    initialX = preProcessor.X

    model = load_facebook_model("vectors/fastText/cc.en.300.bin")

    model.build_vocab(initialX, update=True)
    model.train(initialX, total_examples=len(initialX), epochs=model.epochs)
    vectors = []

    for post in initialX:
        array = np.zeros(300)
        for word in post:
            array += model.wv[word]
        array = np.divide(array, len(post))
        vectors.append(array)

    df = pd.DataFrame(vectors)
    df['Label'] = preProcessor.Y
    stop_words_header = "stopWords"
    lemma_header = "Lemma"
    punct_header = "punct"
    file_name = "fastText"
    if removeStopWords:
        file_name += "-" + stop_words_header
    if useLemmatization:
        file_name += '-' + lemma_header
    if removePunct:
        file_name += '-' + punct_header
    df.to_csv(f'./vectors/fastText/{file_name}.csv')


# To create fastText vectors with Minimal preprocess
createFastTextVectors(removeStopWords=False, useLemmatization=False)


# To create fastText vectors after removing stop words
createFastTextVectors(removeStopWords=True, useLemmatization=False)

#To create fastText vectors after using Lemmaization
createFastTextVectors(removeStopWords=False, useLemmatization=True)

#To create fastText vectors after using Lemmaization and removing stop words
createFastTextVectors(removeStopWords=True, useLemmatization=True)

