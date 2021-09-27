from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from PreProcessor import PreProcessor
import pandas as pd


def createDoc2vecVectors(removeStopWords, useLemmatization, removePunct=False):
    preProcessor = PreProcessor(removeStopWords=removeStopWords, useLemmatization=useLemmatization, removePunct=removePunct)
    preProcessor.splitDbToXandY()
    preProcessor.tokenizeWords()
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
    file_name = "doc2Vec"
    if removeStopWords:
        file_name += "-" + stop_words_header
    if useLemmatization:
        file_name += '-' + lemma_header
    if removePunct:
        file_name += '-' + punct_header

    df.to_csv(f'./vectors/doc2vec/{file_name}.csv')


# To create doc2vec vectors with no preprocess - remove the next comment
createDoc2vecVectors(removeStopWords=False, useLemmatization=False)


#To create doc2vec vectors after removing stop words - remove the next comment
createDoc2vecVectors(removeStopWords=True, useLemmatization=False)

#To create doc2vec vectors after using Lemmaization - remove the next comment
createDoc2vecVectors(removeStopWords=False, useLemmatization=True)

#To create doc2vec vectors after using Lemmaization and removing stop words - remove the next comment
createDoc2vecVectors(removeStopWords=True, useLemmatization=True)

