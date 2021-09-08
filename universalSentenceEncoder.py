import tensorflow_hub as hub
from PreProcessor import PreProcessor
import pandas as pd
import numpy as np



def createUSEVectors(removeStopWords, useLemmatization):
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    preProcessor = PreProcessor(removeStopWords=removeStopWords, useLemmatization=useLemmatization)
    preProcessor.splitDbToXandY()
    preProcessor.cleanPosts()

    initialX = preProcessor.X
    embeddings = embed(initialX)
    embeddings = [obj.numpy() for obj in embeddings]
    df = pd.DataFrame(embeddings)
    df['Label'] = preProcessor.Y
    stop_words_header = "stopWords"
    lemma_header = "Lemma"
    file_name = "USE"
    if removeStopWords:
        file_name += "-"+stop_words_header
    if useLemmatization:
        file_name+='-'+lemma_header

    df.to_csv(f'./vectors/USE/{file_name}.csv')


#To create USE vectores with no preprocess - remove the next comment
#createUSEVectors(removeStopWords=False, useLemmatization=False)

#To create USE vectores after removing stop words - remove the next comment
#createUSEVectors(removeStopWords=True, useLemmatization=False)

#To create USE vectores after using Lemmaization - remove the next comment
#createUSEVectors(removeStopWords=False, useLemmatization=True)

#To create USE vectores after using Lemmaization and removing stop words - remove the next comment
#createUSEVectors(removeStopWords=True, useLemmatization=True)

