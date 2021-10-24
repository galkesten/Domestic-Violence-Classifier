import tensorflow_hub as hub
from PreProcessor import PreProcessor
import pandas as pd

"""
This script creates feature vectors for each post in our DomesticViolence database by using 
Universal Sentence Encoder  model. The  model is pretrained by Google.
The script will create 4 different Universal Sentence Encoder  features due to 4 different pre processing methods.
Minimal Process + Universal Sentence Encoder - vectors will be saved as csv file under this path: vectors/USE/USE.csv
Stop words removal + Universal Sentence Encoder - vectors will be saved as csv file under this path: vectors/USE/USE-stopWords.csv
Lemmatization + Universal Sentence Encoder - vectors will be saved as csv file under this path: vectors/USE/USE-Lemma.csv
Stop words removal + Lemmatization + Universal Sentence Encoder  -vectors will be saved as csv file under this path: vectors/USE/USE-stopWords-Lemma.csv
"""

def createUSEVectors(removeStopWords, useLemmatization, removePunct=False):
    """
        This function creates vector from each post in the DomesticViolence database. The vector is created by using
        the Universal Sentence Encoder model. Before creating the vectors, this function calls to the preprocessor class and all the
        posts in the database are being preprocessed.
        After the vectors creation, the vectors will be saved in csv file under vectors/USE.
        The file name will include the preprocessing method that was used.

        :param removeStopWords: boolean variable that indicates whether or not preform stop words removal
        :param useLemmatization: boolean variable that indicates whether or not preform Lemmatization of each token in each post
        :param removePunct: boolean variable that indicates whether or not preform punctuation removal from each post
        :return: None
    """
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    preProcessor = PreProcessor(removeStopWords=removeStopWords, useLemmatization=useLemmatization, removePunct=removePunct)
    preProcessor.splitDbToXandY()
    preProcessor.getPreProcessedPostsAsStrings()

    initialX = preProcessor.X
    embeddings = embed(initialX)
    embeddings = [obj.numpy() for obj in embeddings]
    df = pd.DataFrame(embeddings)
    df['Label'] = preProcessor.Y
    stop_words_header = "stopWords"
    lemma_header = "Lemma"
    punct_header = "punct"
    file_name = "USE"
    if removeStopWords:
        file_name += "-"+stop_words_header
    if useLemmatization:
        file_name+='-'+lemma_header
    if removePunct:
        file_name += '-' + punct_header

    df.to_csv(f'./vectors/USE/{file_name}.csv')


# To create USE vectors with Minimal preprocess
createUSEVectors(removeStopWords=False, useLemmatization=False)

# To create USE vectors after removing stop words
createUSEVectors(removeStopWords=True, useLemmatization=False)

# To create USE vectors after using Lemmatization
createUSEVectors(removeStopWords=False, useLemmatization=True)

# To create USE vectors after using Lemmatization and removing stop words
createUSEVectors(removeStopWords=True, useLemmatization=True)

