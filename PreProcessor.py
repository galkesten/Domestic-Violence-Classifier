import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class PreProcessor:
    """
        PreProcesser class Preforms different preprocessing methods on the DomesticViolence database.
        Parameters:
            removeStopWords: boolean variable that indicates whether or not preform stop words removal
            useLemmatization: boolean variable that indicates whether or not preform Lemmatization of each token in each post
            removePunct:  boolean variable that indicates whether or not preform punctuation removal from each post

    """
    def __init__(self, removeStopWords=False, useLemmatization=False, removePunct = False):
        self.removeStopWords = removeStopWords
        self.useLemmatization = useLemmatization
        self.removePunct = removePunct

    def splitDbToXandY(self):
        """
        This method loads the DomesticViolence database into preProcessor and divides the
        data to X (which only includes posts)  and Y (which only includes the classes).
        """
        df = pd.read_csv("db/DomesticViolenceDataBase.csv")
        self.X = df['Post']
        self.Y = df['Label']

    def getPreProcessedPostsAsTokens(self):
        """
          This method preforms preProcessing on all the posts that were saved in the class's X variable.
          Each post will be tokenized and will be converted to lower case. additional preproceesing will be applied based
           on the params that were transferred to the class in init function.
          After the preProceesing each post will saved as a list of tokens. The tokenized lists of posts wil be
          saved a dataFrame under the X variable of the class

          """
        stop_words = set(stopwords.words('english'))
        wordnet_lemmatizer = WordNetLemmatizer()
        tokenized_posts = []
        for post in self.X:
            tokenized_post = word_tokenize(post.lower())
            cleaned_post = []
            for word in tokenized_post:
                if self.removeStopWords and word in stop_words:
                    continue
                if self.removePunct and  not word.isalnum():
                    continue
                if self.useLemmatization:
                    lemma = wordnet_lemmatizer.lemmatize(word)
                    cleaned_post.append(lemma)
                else:
                    cleaned_post.append(word)
            tokenized_posts.append(cleaned_post)
        self.X = tokenized_posts

    def getPreProcessedPostsAsStrings(self):
        """
            This method preforms preProcessing on all the posts that were saved in the class's X variable.
            Each post will be tokenized and will be converted to lower case. additional preproceesing will be applied based
            on the params that were transferred to the class in init function.
            After the preProceesing each post will saved as a string. The strings collection will be saved as  a dataFrame
            under the X variable of the class

        """
        stop_words = set(stopwords.words('english'))
        wordnet_lemmatizer = WordNetLemmatizer()
        new_posts = []
        for post in self.X:
            tokenized_post = word_tokenize(post.lower())
            cleaned_post = []
            for word in tokenized_post:
                if self.removeStopWords and word in stop_words:
                    continue
                if self.removePunct and  not word.isalnum():
                    continue
                if self.useLemmatization:
                    lemma = wordnet_lemmatizer.lemmatize(word)
                    cleaned_post.append(lemma)
                else:
                    cleaned_post.append(word)
            new_post = " ".join(cleaned_post)
            new_posts.append(new_post)

        ind = np.arange(len(new_posts))
        df = pd.DataFrame(new_posts, index=ind)
        self.X = df[0]

