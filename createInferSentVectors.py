from models import InferSent
from PreProcessor import PreProcessor
import pandas as pd
import torch
import nltk

"""
This script creates feature vectors for each post in our DomesticViolence database by using 
inferSent model. The inferSent model is pretrained by facebook.
The model name is infersent1 and it needs to be downloaded from here https://github.com/facebookresearch/InferSent#use-our-sentence-encoder.
The model needs to be saved under encoder directory.
Moreover, GloVe vectors also needs to be downloaded from https://github.com/facebookresearch/InferSent#download-word-vectors
and the vectors need to be saved under GloVe directory.

The script will create 4 different infersent features due to 4 different pre processing methods.
Minimal Process + infersent- vectors will be saved as csv file under this path: vectors/inferSent/inferSent.csv
Stop words removal + infersent- vectors will be saved as csv file under this path:vectors/inferSent/inferSent-stopWords.csv  
Lemmatization + infersent- vectors will be saved as csv file under this path: vectors/inferSent/inferSent-Lemma.csv
Stop words removal + Lemmatization + infersent -vectors will be saved as csv file under this path:vectors/inferSent/inferSent-stopWords-Lemma.csv
"""

nltk.download('punkt')
def createInferSentVectors(removeStopWords, useLemmatization, removePunct):
    """
        This function creates vector from each post in the DomesticViolence database. The vector is created by using
        the infersent1 model. Before creating the vectors, this function calls to the preprocessor class and all the
        posts in the database are being preprocessed.
        After the vectors creation, the vectors will be saved in csv file under vectors/inferSent.
        The file name will include the preprocessing method that was used.

        :param removeStopWords: boolean variable that indicates whether or not preform stop words removal
        :param useLemmatization: boolean variable that indicates whether or not preform Lemmatization of each token in each post
        :param removePunct: boolean variable that indicates whether or not preform punctuation removal from each post
        :return: None
    """
    V = 1
    MODEL_PATH = 'encoder/infersent%s.pkl' % V
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    infersent = InferSent(params_model)
    infersent.load_state_dict(torch.load(MODEL_PATH))

    W2V_PATH = 'GloVe/glove.840B.300d.txt'
    infersent.set_w2v_path(W2V_PATH)

    preProcessor = PreProcessor(removeStopWords=removeStopWords, useLemmatization=useLemmatization, removePunct=removePunct)
    preProcessor.splitDbToXandY()
    preProcessor.getPreProcessedPostsAsStrings()

    initialX = preProcessor.X
    infersent.build_vocab(initialX, tokenize=True)
    embeddings = infersent.encode(initialX, tokenize=True)
    df = pd.DataFrame(embeddings)
    df['Label'] = preProcessor.Y
    stop_words_header = "stopWords"
    lemma_header = "Lemma"
    punct_header = "punct"
    file_name = "inferSent"
    if removeStopWords:
        file_name += "-"+stop_words_header
    if useLemmatization:
        file_name+='-'+lemma_header
    if removePunct:
        file_name += '-' + punct_header

    df.to_csv(f'./vectors/inferSent/{file_name}.csv')


# To create inferSent vectors with Minimal preprocess
createInferSentVectors(removeStopWords=False, useLemmatization=False)

# To create inferSenta vectors after removing stop words
createInferSentVectors(removeStopWords=True, useLemmatization=False)

# To create inferSent vectors after using Lemmatization
createInferSentVectors(removeStopWords=False, useLemmatization=True)

# To create inferSent vectors after using Lemmatization and removing stop words
createInferSentVectors(removeStopWords=True, useLemmatization=True)


