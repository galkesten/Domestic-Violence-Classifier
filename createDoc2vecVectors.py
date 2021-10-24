from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from PreProcessor import PreProcessor
import pandas as pd

"""
This script creates feature vectors for each post in our DomesticViolence database by using 
Doc2vec model. The Doc2vec model is implemented by genism package.
The script will create 4 different Doc2vec features due to 4 different pre processing methods.
Minimal Process + Doc2vec- vectors will be saved as csv file under this path: ./vectors/doc2vec/doc2vec.csv
Stop words removal +Doc2vec- vectors will be saved as csv file under this path: vectors/doc2vec/doc2vec-stopWords.csv  
Lemmatization + Doc2vec- vectors will be saved as csv file under this path: vectors/doc2vec/doc2vec-Lemma.csv
Stop words removal + Lemmatization + Doc2vec -vectors will be saved as csv file under this path: vectors/doc2vec/doc2vec-stopWords-Lemma.csv
"""


def createDoc2vecVectors(removeStopWords, useLemmatization, removePunct=False):
    """
    This function creates vector from each post in the DomesticViolence database. The vector is created by using
    the doc2vec model. Before creating the vectors, this function calls to the preprocessor class and all the
    posts in the database are being preprocessed.
    After the vectors creation, the vectors will be saved in csv file under vectors/doc2vec.
    The file name will include the preprocessing method that was used.

    :param removeStopWords: boolean variable that indicates whether or not preform stop words removal
    :param useLemmatization: boolean variable that indicates whether or not preform Lemmatization of each token in each post
    :param removePunct: boolean variable that indicates whether or not preform punctuation removal from each post
    :return: None
    """
    preProcessor = PreProcessor(removeStopWords=removeStopWords, useLemmatization=useLemmatization, removePunct=removePunct)
    preProcessor.splitDbToXandY()
    preProcessor.getPreProcessedPostsAsTokens()
    initialX = preProcessor.X

    # Convert tokenized document into gensim formated tagged data
    tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(initialX)]
    X = []
    # Train doc2vec model
    model = Doc2Vec(tagged_data, vector_size=300, min_count=2, epochs=40)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    for doc_id in range(len(tagged_data)):
        inferred_vector = model.infer_vector(tagged_data[doc_id].words)
        X.append(inferred_vector)

    df = pd.DataFrame(X)
    df['Label'] = preProcessor.Y
    stop_words_header = "stopWords"
    lemma_header = "Lemma"
    punct_header = "punct"
    file_name = "doc2vec"
    if removeStopWords:
        file_name += "-" + stop_words_header
    if useLemmatization:
        file_name += '-' + lemma_header
    if removePunct:
        file_name += '-' + punct_header

    df.to_csv(f'./vectors/doc2vec/{file_name}.csv')


# To create doc2vec vectors with Minimal preprocess
createDoc2vecVectors(removeStopWords=False, useLemmatization=False)


# To create doc2vec vectors after removing stop words
createDoc2vecVectors(removeStopWords=True, useLemmatization=False)

# To create doc2vec vectors after using Lemmatization
createDoc2vecVectors(removeStopWords=False, useLemmatization=True)

# To create doc2vec vectors after using Lemmatization and removing stop words
createDoc2vecVectors(removeStopWords=True, useLemmatization=True)

