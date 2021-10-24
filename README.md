# Files List

1. **creatDatabase.py**-This is a script  which creates a combined DomesticViolence database from the negative posts database (not critical posts) the positive posts database (critical posts). All the databases are saved under db directory.
2. **createBagOfWordsVectors.py**- This script creates feature vectors for each post in our DomesticViolence database by using 
Bag of words model. The Bag of words model is implemented by sklearn package.
The script will create 4 different Bag of words features due to 4 different pre processing methods. The feature vectors will be saved in a csv file format in vectors/BagOfWords.
3. **createDoc2vecVectors.py**- This scripts creates feature vectors for each post in our DomesticViolence database by using 
Doc2vec model. The Doc2vec model is implemented by genism package.The feature vectors will be saved in a csv file format in vectors/doc2vec.

4. **createFastTextVectors.py**- This script creates feature vectors for each post in our DomesticViolence database by using fastText model. The fast text model is pre trained and in order to run you need to download the file cc.en.300.bin.gz
from https://fasttext.cc/docs/en/crawl-vectors.html and save it under vectors/fastText/cc.en.300.bin.gz.
The feature vectors will be saved in a csv file format in vectors/fastText.

5. **createInferSentVectors.py** - This scripts creates feature vectors for each post in our DomesticViolence database by using 
inferSent model. The inferSent model is pretrained by facebook.
The model name is infersent1 and it needs to be downloaded from here https://github.com/facebookresearch/InferSent#use-our-sentence-encoder .
The model needs to be saved under encoder directory.
Moreover, GloVe vectors also needs to be downloaded from https://github.com/facebookresearch/InferSent#download-word-vectors
and the vectors need to be saved under GloVe directory.
The feature vectors will be saved in a csv file format in vectors/inferSent.
6. **createPartOfSpeechVectors.py**- This script creates syntactic feature vectors for each post in our DomesticViolence database by using 
BagOfPos and POSWithTFIDF model.
The script will create 4 different kind of features for each model due to 4 different pre processing methods.
The feature vectors will be saved in a csv file format in vectors/BagOfPOS and 
vectors/POSWithTfidf.

7.**createSentenceTransformersVectors.py**- This script creates feature vectors for each post in our DomesticViolence database by using 
pretraind models from sentence_transformers package.
The script will create 4 different kind of features for each model due to 4 different pre processing methods.
The feature vectors will be saved in a csv file format in vectors/BagOfPOS and 
vectors/MiniLM, vectors/MPNet, vectors/Roberta.

8.**createTfidfVecrors.py**- This function creates vector from each post in the DomesticViolence database. The vector is created by using
the TFIDF model. Before creating the vectors, this function calls to the preprocessor class and all the
posts in the database are being preprocessed. The feature vectors will be saved in a csv file format in 
vectors/TFIDF.

9.**createUniversalSentenceEncoderVectors.py**- This script creates feature vectors for each post in our DomesticViolence database by using 
Universal Sentence Encoder  model. The  model is pretrained by Google.
The feature vectors will be saved in a csv file format in vectors/USE.

10. **Experiments.py**- This script run the Experiments we conducted during the project.
The script will ask you which number of Experiment you want to run.
It expects to get one of these numbers as input:
1 / 2 / 3 / 0. 
0 - for running all the experiments.
For each Experiment, A csv file with all the results will be created under The Results Directory.

11. **finalClassifier.py**- This script run the final classifier that wad described in our paper.
The script will ask you to type a post as an input, ant it will print its classification as an output.

12. **LearningAlgorithms.py** -
This file contain functions that train different classifiers with StratifiedKFold with K=5.
It also contain calcAccuracy function which calculates the average accuracy of different classifiers that are trained on X featuers. The average accuracy for each classifier is than printed to a csv file.
The name of the file will be in this format- Experiment{numExperiment}-results-{dateStr}.csv snd will be saved
in Results directory.
Each time this function called a new row will be added to the csv file and the accuracies will be printed in the appropriate column.
13. **models.py** - 
This file contains the definition of encoders used in https://arxiv.org/pdf/1705.02364.pdf. The file was downloaded from here https://github.com/facebookresearch/InferSent#use-our-sentence-encoder. 
14. PreProcessor.py- PreProcesser class Preforms different preprocessing methods on the DomesticViolence database..

# Directories list
**db**- contain our DomesticViolenceDataBase.csv.
In addition it containd different csv files with the examples we used, seprated to negative examples and positivie ones.
**encoder***- This is a directory to save the infersent1 model.
Should be downloaded from here:
https://github.com/facebookresearch/InferSent#use-our-sentence-encoder

**Glove**- A directory to save Glove word vectors.
Should be downloaded from here:
https://github.com/facebookresearch/InferSent#download-word-vectors

**Results**- contain all the results from the experiments. After running Experiments script, the results will be saved in this directory.

**vectors** - contain different sub directories for each type we features we tried in the experiments. The sub directories contain csv file with vector representaion for each example.
