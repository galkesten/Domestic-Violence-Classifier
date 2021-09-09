import pandas as pd
import numpy as np
from nltk.tokenize import  word_tokenize
from nltk.corpus import stopwords
from nltk.stem import 	WordNetLemmatizer

class PreProcessor:
    def __init__(self, removeStopWords, useLemmatization):
        self.removeStopWords = removeStopWords
        self.useLemmatization = useLemmatization

    def splitDbToXandY(self):
        df = pd.read_csv("db/DomecticViolence.csv")
        self.X = df['Post']
        self.Y = df['Label']

    def tokenizeWords(self):
        stop_words = set(stopwords.words('english'))
        #print(stop_words)
        wordnet_lemmatizer = WordNetLemmatizer()
        tokenized_posts = []
        for post in self.X:
            tokenized_post = word_tokenize(post.lower())
            cleaned_post = []
            for word in tokenized_post:
                if self.removeStopWords and word in stop_words:
                    continue
                if not word.isalnum():
                    continue
                if self.useLemmatization:
                    lemma = wordnet_lemmatizer.lemmatize(word)
                    #if lemma != word:
                     #   print("Lemma for {} is {}".format(word, lemma))
                    cleaned_post.append(lemma)
                else:
                    cleaned_post.append(word)
            tokenized_posts.append(cleaned_post)
            #print( cleaned_post)
        self.X = tokenized_posts

    def cleanPosts(self):
        stop_words = set(stopwords.words('english'))
        wordnet_lemmatizer = WordNetLemmatizer()
        new_posts = []
        for post in self.X:
            tokenized_post = word_tokenize(post.lower())
            cleaned_post = []
            for word in tokenized_post:
                if self.removeStopWords and word in stop_words:
                    continue
                if not word.isalnum():
                    continue
                if self.useLemmatization:
                    lemma = wordnet_lemmatizer.lemmatize(word)
                    #if lemma != word:
                     #   print("Lemma for {} is {}".format(word, lemma))
                    cleaned_post.append(lemma)
                else:
                    cleaned_post.append(word)
            new_post = " ".join(cleaned_post)
            new_posts.append(new_post)

        ind = np.arange(len(new_posts))
        df = pd.DataFrame(new_posts, index=ind)
        self.X = df[0]

