import gensim.downloader as api
from nltk.tokenize import  word_tokenize
import pandas as pd
import numpy as np

df = pd.read_csv("DomecticViolence.csv")
wordToVec = api.load('word2vec-google-news-300')


tokenized_posts = []
for post in df['Post']:
    tokenized_post = word_tokenize(post.lower())
    cleaned_post = [word for word in tokenized_post if word.isalnum()]
    for word in cleaned_post:
        try:
            vec_cameroon = wordToVec[word]
        except KeyError:
            print(word)

