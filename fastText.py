from PreProcessor import PreProcessor
from gensim.test.utils import datapath
from gensim.models.fasttext import load_facebook_vectors, load_facebook_model
import numpy as np
import pandas as pd

def createFastTextVectors(removeStopWords, useLemmatization, removePunct=False):
    preProcessor = PreProcessor(removeStopWords=removeStopWords, useLemmatization=useLemmatization, removePunct=removePunct)
    preProcessor.splitDbToXandY()
    preProcessor.tokenizeWords()
    initialX = preProcessor.X

    #dataPath = datapath("wiki-news-300d-1M.vec")
    #model = load_facebook_vectors("./trainedModels/cc.en.300.bin")
    model = load_facebook_model("./trainedModels/cc.en.300.bin")

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


# To create fastText vectors with no preprocess - remove the next comment
#createFastTextVectors(removeStopWords=False, useLemmatization=False)


#To create fastText vectors after removing stop words - remove the next comment
#createFastTextVectors(removeStopWords=True, useLemmatization=False)

#To create fastText vectors after using Lemmaization - remove the next comment
#createFastTextVectors(removeStopWords=False, useLemmatization=True)

#To create fastText vectors after using Lemmaization and removing stop words - remove the next comment
#createFastTextVectors(removeStopWords=True, useLemmatization=True)

#To create vectors for punct experiment- remove the next comment
createFastTextVectors(removeStopWords=False, useLemmatization=False, removePunct=True)
createFastTextVectors(removeStopWords=True, useLemmatization=True, removePunct=True)
createFastTextVectors(removeStopWords=True, useLemmatization=False, removePunct=True)