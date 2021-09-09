from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from PreProcessor import PreProcessor
from run_algo import get_accuracy
import run_algo
import pandas as pd
import numpy as np

def getXandY(filePath, useTfidf):
    df = pd.read_csv(filePath)
    Y = np.array(df['Label'])
    df = df.drop(columns=['Label'])
    X = df.to_numpy()
    X = np.delete(X, 0, 1)
    if not useTfidf:
        return X,Y
    preProcessor = PreProcessor(False, False)
    preProcessor.splitDbToXandY()
    vectorizer = TfidfVectorizer()
    tfidfVectors = vectorizer.fit_transform(preProcessor.X)
    tfidfVectors = tfidfVectors.toarray()
    X = np.hstack((X, tfidfVectors))
    return X,Y

def bagOfWordsExperiment():
    preProcessor = PreProcessor(False, False)
    preProcessor.splitDbToXandY()
    header = "BagOfWords"
    #with no preprocessing
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(preProcessor.X)
    X = X.toarray()
    Y = preProcessor.Y

    get_accuracy(X, Y, header+ "-no preprocess")

    # with stop words preprocessing
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(preProcessor.X)
    X = X.toarray()
    Y = preProcessor.Y

    get_accuracy(X, Y, header+"-stop words removal")

    #with Lemmatization
    preProcessor = PreProcessor(False, True)
    preProcessor.splitDbToXandY()
    preProcessor.cleanPosts()

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(preProcessor.X)
    X = X.toarray()
    Y = preProcessor.Y

    get_accuracy(X, Y, header+"-Lemmatization")

    #with stop words removal and Lemmatization
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(preProcessor.X)
    X = X.toarray()
    Y = preProcessor.Y
    get_accuracy(X, Y, header+"-Lemmatization + stop words removal")

def tfidfExperiment():
    preProcessor = PreProcessor(False, False)
    preProcessor.splitDbToXandY()
    header = "tfidf"

    #with no preprocessing
    vectorizer =  TfidfVectorizer()
    X = vectorizer.fit_transform(preProcessor.X)
    X = X.toarray()
    Y = preProcessor.Y

    get_accuracy(X, Y, header+"-no preprocess")

    # with stop words preprocessing
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(preProcessor.X)
    X = X.toarray()
    Y = preProcessor.Y

    get_accuracy(X, Y, header+"-stop words removal")

    #with Lemmatization
    preProcessor = PreProcessor(False, True)
    preProcessor.splitDbToXandY()
    preProcessor.cleanPosts()

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(preProcessor.X)
    X = X.toarray()
    Y = preProcessor.Y

    get_accuracy(X, Y, header+"-Lemmatization")

    #with stop words removal and Lemmatization
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(preProcessor.X)
    X = X.toarray()
    Y = preProcessor.Y
    get_accuracy(X, Y, header+"-Lemmatization + stop words removal")


def doc2vecExperiment(useTfidf = False):
   #no preprocess
   header = "doc2vec"
   if useTfidf:
       header+= "+tfidf"
   X,Y = getXandY("./vectors/doc2vec/doc2Vec.csv", useTfidf)
   get_accuracy(X,Y, header+ "-no preprocess")

   #with stop words removal
   X, Y = getXandY("./vectors/doc2vec/doc2Vec-stopWords.csv", useTfidf)
   get_accuracy(X, Y, header+ "-stop words removal")

   #with Lemmatization
   X,Y =  getXandY("./vectors/doc2vec/doc2Vec-Lemma.csv",useTfidf)
   get_accuracy(X, Y, header+ "-Lemmatization")


   #with stop words removal and Lemmatization
   X, Y = getXandY("./vectors/doc2vec/doc2Vec-stopWords-Lemma.csv", useTfidf)
   get_accuracy(X, Y, header+"-Lemmatization + stop words removal")

def fastTextExperiment(useTfidf = False):
   #no preprocess
   header = "fastText"
   if useTfidf:
       header+= "+tfidf"
   X,Y = getXandY("./vectors/fastText/fastText.csv", useTfidf)
   get_accuracy(X,Y, header+ "-no preprocess")

   #with stop words removal
   X, Y = getXandY("./vectors/fastText/fastText-stopWords.csv", useTfidf)
   get_accuracy(X, Y, header+ "-stop words removal")

   #with Lemmatization
   X,Y =  getXandY("./vectors/fastText/fastText-Lemma.csv", useTfidf)
   get_accuracy(X, Y, header + "-Lemmatization")


   #with stop words removal and Lemmatization
   X, Y = getXandY("./vectors/fastText/fastText-stopWords-Lemma.csv", useTfidf)
   get_accuracy(X, Y, header+"-Lemmatization + stop words removal")

def inferSentExperiment(useTfidf = False):
   #no preprocess
   header = "inferSent"
   if useTfidf:
       header+= "+tfidf"
   X,Y = getXandY("./vectors/inferSent/inferSent.csv", useTfidf)
   get_accuracy(X,Y, header+ "-no preprocess")

   #with stop words removal
   X, Y = getXandY("./vectors/inferSent/inferSent-stopWords.csv", useTfidf)
   get_accuracy(X, Y, header+ "-stop words removal")

   #with Lemmatization
   X,Y =  getXandY("./vectors/inferSent/inferSent-Lemma.csv", useTfidf)
   get_accuracy(X, Y, header + "-Lemmatization")


   #with stop words removal and Lemmatization
   X, Y = getXandY("./vectors/inferSent/inferSent-stopWords-Lemma.csv", useTfidf)
   get_accuracy(X, Y, header+"-Lemmatization + stop words removal")

def miniLMExperiment(useTfidf = False):
   #no preprocess
   header = "bert-miniLM"
   if useTfidf:
       header+= "+tfidf"
   X,Y = getXandY("./vectors/MiniLM/miniLM.csv", useTfidf)
   get_accuracy(X,Y, header+ "-no preprocess")

   #with stop words removal
   X, Y = getXandY("./vectors/MiniLM/miniLM-stopWords.csv", useTfidf)
   get_accuracy(X, Y, header+ "-stop words removal")

   #with Lemmatization
   X,Y =  getXandY("./vectors/MiniLM/miniLM-Lemma.csv", useTfidf)
   get_accuracy(X, Y, header + "-Lemmatization")


   #with stop words removal and Lemmatization
   X, Y = getXandY("./vectors/MiniLM/miniLM-stopWords-Lemma.csv", useTfidf)
   get_accuracy(X, Y, header+"-Lemmatization + stop words removal")

def mpnetExperiment(useTfidf = False):
   #no preprocess
   header = "bert-mpnet"
   if useTfidf:
       header+= "+tfidf"
   X,Y = getXandY("./vectors/mpnet/mpnet.csv", useTfidf)
   get_accuracy(X,Y, header+ "-no preprocess")

   #with stop words removal
   X, Y = getXandY("./vectors/mpnet/mpnet-stopWords.csv", useTfidf)
   get_accuracy(X, Y, header+ "-stop words removal")

   #with Lemmatization
   X,Y =  getXandY("./vectors/mpnet/mpnet-Lemma.csv", useTfidf)
   get_accuracy(X, Y, header + "-Lemmatization")


   #with stop words removal and Lemmatization
   X, Y = getXandY("./vectors/mpnet/mpnet-stopWords-Lemma.csv", useTfidf)
   get_accuracy(X, Y, header+"-Lemmatization + stop words removal")

def robertaExperiment(useTfidf = False):
   #no preprocess
   header = "bert-roberta"
   if useTfidf:
       header+= "+tfidf"
   X,Y = getXandY("./vectors/Roberta/roberta.csv", useTfidf)
   get_accuracy(X,Y, header+ "-no preprocess")

   #with stop words removal
   X, Y = getXandY("./vectors/Roberta/roberta-stopWords.csv", useTfidf)
   get_accuracy(X, Y, header+ "-stop words removal")

   #with Lemmatization
   X,Y =  getXandY("./vectors/Roberta/roberta-Lemma.csv", useTfidf)
   get_accuracy(X, Y, header + "-Lemmatization")


   #with stop words removal and Lemmatization
   X, Y = getXandY("./vectors/Roberta/roberta-stopWords-Lemma.csv", useTfidf)
   get_accuracy(X, Y, header+"-Lemmatization + stop words removal")

def USEExperiment(useTfidf = False):
   #no preprocess
   header = "USE"
   if useTfidf:
       header+= "+tfidf"
   X,Y = getXandY("./vectors/USE/USE.csv", useTfidf)
   get_accuracy(X,Y, header+ "-no preprocess")

   #with stop words removal
   X, Y = getXandY("./vectors/USE/USE-stopWords.csv", useTfidf)
   get_accuracy(X, Y, header+ "-stop words removal")

   #with Lemmatization
   X,Y =  getXandY("./vectors/USE/USE-Lemma.csv", useTfidf)
   get_accuracy(X, Y, header + "-Lemmatization")


   #with stop words removal and Lemmatization
   X, Y = getXandY("./vectors/USE/USE-stopWords-Lemma.csv", useTfidf)
   get_accuracy(X, Y, header+"-Lemmatization + stop words removal")

#bagOfWordsExperiment()
#tfidfExperiment()
#fastTextExperiment()
#fastTextExperiment(useTfidf=True)
#doc2vecExperiment()
#doc2vecExperiment(useTfidf=True)
#inferSentExperiment()
#inferSentExperiment(useTfidf=True)
#miniLMExperiment()
#miniLMExperiment(useTfidf=True)
#mpnetExperiment()
#miniLMExperiment(useTfidf=True)
#robertaExperiment()
#robertaExperiment(useTfidf=True)
#USEExperiment()
#USEExperiment(useTfidf=True)
