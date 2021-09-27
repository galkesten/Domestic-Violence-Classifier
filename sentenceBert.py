from sentence_transformers import SentenceTransformer
from PreProcessor import PreProcessor
import pandas as pd
import numpy as np
import os

def createRobertaVectors(removeStopWords, useLemmatization, removePunct=False):
    bert = SentenceTransformer('all-distilroberta-v1')
    print("Max Sequence Length:", bert.max_seq_length)
    bert.max_seq_length = 512
    preProcessor = PreProcessor(removeStopWords=removeStopWords, useLemmatization=useLemmatization, removePunct=removePunct)
    preProcessor.splitDbToXandY()
    preProcessor.cleanPosts()

    initialX = preProcessor.X
    df = pd.DataFrame(np.vstack(initialX.apply(bert.encode)))
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
    bert = SentenceTransformer('all-MiniLM-L6-v2')
    print("Max Sequence Length:", bert.max_seq_length)
    bert.max_seq_length = 512
    preProcessor = PreProcessor(removeStopWords=removeStopWords, useLemmatization=useLemmatization, removePunct=removePunct)
    preProcessor.splitDbToXandY()
    preProcessor.cleanPosts()

    initialX = preProcessor.X
    df = pd.DataFrame(np.vstack(initialX.apply(bert.encode)))
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
    bert = SentenceTransformer('all-mpnet-base-v2')
    print("Max Sequence Length:", bert.max_seq_length)
    bert.max_seq_length = 512
    preProcessor = PreProcessor(removeStopWords=removeStopWords, useLemmatization=useLemmatization, removePunct=removePunct)
    preProcessor.splitDbToXandY()
    preProcessor.cleanPosts()

    initialX = preProcessor.X
    df = pd.DataFrame(np.vstack(initialX.apply(bert.encode)))
    df['Label'] = preProcessor.Y
    stop_words_header = "stopWords"
    lemma_header = "Lemma"
    punct_header = "punct"
    file_name = "mpnet"
    if removeStopWords:
        file_name += "-"+stop_words_header
    if useLemmatization:
        file_name+='-'+lemma_header
    if removePunct:
        file_name += '-' + punct_header

    df.to_csv(f'./vectors/mpnet/{file_name}.csv')

#To create roberta vectores with no preprocess - remove the next comment
#createRobertaVectors(removeStopWords=False, useLemmatization=False)

#To create roberta vectores after removing stop words - remove the next comment
#createRobertaVectors(removeStopWords=True, useLemmatization=False)

#To create roberta vectores after using Lemmaization - remove the next comment
#createRobertaVectors(removeStopWords=False, useLemmatization=True)

#To create roberta vectores after using Lemmaization and removing stop words - remove the next comment
#createRobertaVectors(removeStopWords=True, useLemmatization=True)

#To create MiniLM vectores with no preprocess - remove the next comment
#createMiniLMVectors(removeStopWords=False, useLemmatization=False)

#To create MiniLM vectores after removing stop words - remove the next comment
#createMiniLMVectors(removeStopWords=True, useLemmatization=False)

#To create MiniLM vectores after using Lemmaization - remove the next comment
#createMiniLMVectors(removeStopWords=False, useLemmatization=True)

#To create MiniLM vectores after using Lemmaization and removing stop words - remove the next comment
#createMiniLMVectors(removeStopWords=True, useLemmatization=True)


#To create mpnet vectores with no preprocess - remove the next comment
#createMpnetVectors(removeStopWords=False, useLemmatization=False)

#To create mpnet vectores after removing stop words - remove the next comment
#createMpnetVectors(removeStopWords=True, useLemmatization=False)

#To create mpnet vectores after using Lemmaization - remove the next comment
#createMpnetVectors(removeStopWords=False, useLemmatization=True)

#To create mpnet vectores after using Lemmaization and removing stop words - remove the next comment
#createMpnetVectors(removeStopWords=True, useLemmatization=True)


#To create vectors for punct experiment- remove the next comment
createRobertaVectors(removeStopWords=False, useLemmatization=False, removePunct=True)
createMpnetVectors(removeStopWords=False, useLemmatization=True, removePunct=True)