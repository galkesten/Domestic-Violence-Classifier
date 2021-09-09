from models import InferSent
from PreProcessor import PreProcessor
import pandas as pd
import torch

import nltk
nltk.download('punkt')
def createInferSentVectors(removeStopWords, useLemmatization):
    V = 1
    MODEL_PATH = 'encoder/infersent%s.pkl' % V
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    infersent = InferSent(params_model)
    infersent.load_state_dict(torch.load(MODEL_PATH))

    W2V_PATH = 'GloVe/glove.840B.300d.txt'
    infersent.set_w2v_path(W2V_PATH)

    preProcessor = PreProcessor(removeStopWords=removeStopWords, useLemmatization=useLemmatization)
    preProcessor.splitDbToXandY()
    preProcessor.cleanPosts()

    initialX = preProcessor.X
    infersent.build_vocab(initialX, tokenize=True)
    embeddings = infersent.encode(initialX, tokenize=True)
    df = pd.DataFrame(embeddings)
    df['Label'] = preProcessor.Y
    stop_words_header = "stopWords"
    lemma_header = "Lemma"
    file_name = "inferSent"
    if removeStopWords:
        file_name += "-"+stop_words_header
    if useLemmatization:
        file_name+='-'+lemma_header

    df.to_csv(f'./vectors/inferSent/{file_name}.csv')


#To create inferSent vectores with no preprocess - remove the next comment
#createInferSentVectors(removeStopWords=False, useLemmatization=False)

#To create roberta vectores after removing stop words - remove the next comment
createInferSentVectors(removeStopWords=True, useLemmatization=False)

#To create roberta vectores after using Lemmaization - remove the next comment
createInferSentVectors(removeStopWords=False, useLemmatization=True)

#To create roberta vectores after using Lemmaization and removing stop words - remove the next comment
createInferSentVectors(removeStopWords=True, useLemmatization=True)


