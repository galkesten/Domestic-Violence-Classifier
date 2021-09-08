import gensim.downloader as api
from nltk.tokenize import  word_tokenize
import pandas as pd
import numpy as np
from PreProcessor import PreProcessor

preProcessor = PreProcessor(False, False)
preProcessor.splitDbToXandY()
preProcessor.tokenizeWords()

initialX = preProcessor.X
wordToVec = api.load('word2vec-google-news-300')
wordToVec.build_vocab(initialX, update=True)
wordToVec.train(initialX, total_examples=wordToVec.corpus_count, epochs=wordToVec.epochs)

vectors = []

for post in initialX:
    array = np.zeros(300)
    for word in post:
        try:
            vec = wordToVec[word]
        except KeyError:
            continue
        array += vec
    array = array / len(post)
    vectors.append(array)

df = pd.DataFrame(vectors)
df['Label'] = preProcessor.Y
df.to_csv("wordToVec.csv")
