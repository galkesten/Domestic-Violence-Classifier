from sentence_transformers import SentenceTransformer
from PreProcessor import PreProcessor
import pandas as pd
import numpy as np

"""
This script creates  feature vectors for each post in our DomesticViolence database by using 
pretraind models from sentence_transformers package.
The script will create 4 different kind of features for each model due to 4 different pre processing methods.
For DistilRoBERTa:
Minimal Process + DistilRoBERTa- vectors will be saved as csv file under this path: vectors/Roberta/roberta.csv
Stop words removal + DistilRoBERTa- vectors will be saved as csv file under this path: vectors/Roberta/roberta-stopWords.csv
Punctuation removal + DistilRoBERTa- vectors will be saved as csv file under this path:vectors/Roberta/roberta-Lemma.csv
Stop words removal + Punctuation removal +  DistilRoBERTa -vectors will be saved as csv file under this path: vectors/Roberta/roberta-stopWords-Lemma.csv

For MPNet:
Minimal Process + MPNet- vectors will be saved as csv file under this path: vectors/MPNet/mpnet.csv
Stop words removal + MPNet- vectors will be saved as csv file under this path: vectors/MPNet/mpnet-stopWords.csv
Punctuation removal + MPNet- vectors will be saved as csv file under this path: vectors/MPNet/mpnet-Lemma.csv
Stop words removal + Punctuation removal +MPNet-vectors will be saved as csv file under this path: vectors/MPNet/mpnet-stopWords-Lemma.csv

For MiniLM:
Minimal Process + MiniLM- vectors will be saved as csv file under this path: vectors/MiniLM/miniLM.csv
Stop words removal + MiniLM- vectors will be saved as csv file under this path: vectors/MiniLM/miniLM-stopWords.csv
Punctuation removal +MiniLM- vectors will be saved as csv file under this path: vvectors/MiniLM/miniLM-Lemma.csv
Stop words removal + Punctuation removall +MiniLM-vectors will be saved as csv file under this path: vectors/MiniLM/miniLM-stopWords-Lemma.csv

"""

def createRobertaVectors(removeStopWords, useLemmatization, removePunct=False):
    """
    This function creates vector from each post in the DomesticViolence database. The vector is created by using
    the DistilRoBERTa model. Before creating the vectors, this function calls to the preprocessor class and all the
    posts in the database are being preprocessed.
    After the vectors creation, the vectors will be saved in csv file under vectors/Roberta.
    The file name will include the preprocessing method that was used.

    :param removeStopWords: boolean variable that indicates whether or not preform stop words removal
    :param useLemmatization: boolean variable that indicates whether or not preform Lemmatization of each token in each post
    :param removePunct: boolean variable that indicates whether or not preform punctuation removal from each post
    :return: None
    """
    roberta = SentenceTransformer('all-distilroberta-v1')
    print("Max Sequence Length:", roberta.max_seq_length)
    roberta.max_seq_length = 512
    preProcessor = PreProcessor(removeStopWords=removeStopWords, useLemmatization=useLemmatization, removePunct=removePunct)
    preProcessor.splitDbToXandY()
    preProcessor.getPreProcessedPostsAsStrings()

    initialX = preProcessor.X
    df = pd.DataFrame(np.vstack(initialX.apply(roberta.encode)))
    df['Label'] = preProcessor.Y
    stop_words_header = "stopWords"
    lemma_header = "Lemma"
    punct_header = "punct"
    file_name = "roberta"
    if removeStopWords:
        file_name += "-"+stop_words_header
    if useLemmatization:
        file_name+='-'+lemma_header
    if removePunct:
        file_name += '-' + punct_header

    df.to_csv(f'./vectors/Roberta/{file_name}.csv')


def createMiniLMVectors(removeStopWords, useLemmatization, removePunct):
    """
       This function creates vector from each post in the DomesticViolence database. The vector is created by using
       the MiniLM model. Before creating the vectors, this function calls to the preprocessor class and all the
       posts in the database are being preprocessed.
       After the vectors creation, the vectors will be saved in csv file under vectors/MiniLM.
       The file name will include the preprocessing method that was used.

       :param removeStopWords: boolean variable that indicates whether or not preform stop words removal
       :param useLemmatization: boolean variable that indicates whether or not preform Lemmatization of each token in each post
       :param removePunct: boolean variable that indicates whether or not preform punctuation removal from each post
       :return: None
       """
    miniLM = SentenceTransformer('all-MiniLM-L6-v2')
    print("Max Sequence Length:", miniLM.max_seq_length)
    miniLM.max_seq_length = 512
    preProcessor = PreProcessor(removeStopWords=removeStopWords, useLemmatization=useLemmatization, removePunct=removePunct)
    preProcessor.splitDbToXandY()
    preProcessor.getPreProcessedPostsAsStrings()

    initialX = preProcessor.X
    df = pd.DataFrame(np.vstack(initialX.apply(miniLM.encode)))
    df['Label'] = preProcessor.Y
    stop_words_header = "stopWords"
    lemma_header = "Lemma"
    punct_header = "punct"
    file_name = "miniLM"
    if removeStopWords:
        file_name += "-"+stop_words_header
    if useLemmatization:
        file_name+='-'+lemma_header
    if removePunct:
        file_name += '-' + punct_header

    df.to_csv(f'./vectors/MiniLm/{file_name}.csv')

def createMpnetVectors(removeStopWords, useLemmatization, removePunct):
    """
       This function creates vector from each post in the DomesticViolence database. The vector is created by using
       the MPNet model. Before creating the vectors, this function calls to the preprocessor class and all the
       posts in the database are being preprocessed.
       After the vectors creation, the vectors will be saved in csv file under vectors/MPNet.
       The file name will include the preprocessing method that was used.

       :param removeStopWords: boolean variable that indicates whether or not preform stop words removal
       :param useLemmatization: boolean variable that indicates whether or not preform Lemmatization of each token in each post
       :param removePunct: boolean variable that indicates whether or not preform punctuation removal from each post
       :return: None
       """
    mpnet = SentenceTransformer('all-MPNet-base-v2')
    print("Max Sequence Length:", mpnet.max_seq_length)
    mpnet.max_seq_length = 512
    preProcessor = PreProcessor(removeStopWords=removeStopWords, useLemmatization=useLemmatization, removePunct=removePunct)
    preProcessor.splitDbToXandY()
    preProcessor.getPreProcessedPostsAsStrings()

    initialX = preProcessor.X
    df = pd.DataFrame(np.vstack(initialX.apply(mpnet.encode)))
    df['Label'] = preProcessor.Y
    stop_words_header = "stopWords"
    lemma_header = "Lemma"
    punct_header = "punct"
    file_name = "MPNet"
    if removeStopWords:
        file_name += "-"+stop_words_header
    if useLemmatization:
        file_name+='-'+lemma_header
    if removePunct:
        file_name += '-' + punct_header

    df.to_csv(f'./vectors/MPNet/{file_name}.csv')

# To create roberta vectors with Minimal preprocess
createRobertaVectors(removeStopWords=False, useLemmatization=False)

# To create roberta vectors after removing stop words
createRobertaVectors(removeStopWords=True, useLemmatization=False)

# To create roberta vectors after using Lemmatization
createRobertaVectors(removeStopWords=False, useLemmatization=True)

# To create roberta vectors after using Lemmatization and removing stop words
createRobertaVectors(removeStopWords=True, useLemmatization=True)

# To create MiniLM vectors with Minimal preprocess
createMiniLMVectors(removeStopWords=False, useLemmatization=False)

# To create MiniLM vectors after removing stop words
createMiniLMVectors(removeStopWords=True, useLemmatization=False)

# To create MiniLM vectors after using Lemmatization
createMiniLMVectors(removeStopWords=False, useLemmatization=True)

# To create MiniLM vectors after using Lemmatization and removing stop words
createMiniLMVectors(removeStopWords=True, useLemmatization=True)


# To create MPNet vectors with Minimal preprocess
createMpnetVectors(removeStopWords=False, useLemmatization=False)

# To create MPNet vectors after removing stop words
createMpnetVectors(removeStopWords=True, useLemmatization=False)

# To create MPNet vectors after using Lemmatization
createMpnetVectors(removeStopWords=False, useLemmatization=True)

# To create MPNet vectors after using Lemmatization and removing stop words
createMpnetVectors(removeStopWords=True, useLemmatization=True)


