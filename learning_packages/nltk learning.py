from nltk.tokenize import  word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import 	WordNetLemmatizer

from nltk.corpus import stopwords
import pandas as pd
text="Hello there! Welcome to the programming world."
text= word_tokenize(text)
new_words= [word for word in text if word.isalnum()]

print(new_words)

df = pd.read_csv("DomecticViolence.csv")
df['tokenized'] = df.apply(lambda row: word_tokenize(row['Post']), axis=1)
print(df.head())
df.to_csv("DomesticViolenceTokenized.csv", index=False)


#stemming- we are trying to reduce this words to a root word

s_stremmer = SnowballStemmer(language='english')
words = ['run', 'runner', 'ran', 'runs', 'easily', 'fairly', 'fairness']

for word in words:
    print(word + '----->'+ s_stremmer.stem(word))

#stop words

example_sent = """This is a sample sentence,
                  showing off the stop words filtration."""

stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(example_sent)

filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
print(filtered_sentence)
filtered_sentence = []

for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)

print(word_tokens)
print(filtered_sentence)

#Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()
text = "studies studying cries cry"
tokenization =word_tokenize(text)
for w in tokenization:
    print("Lemma for {} is {}".format(w, wordnet_lemmatizer.lemmatize(w)))