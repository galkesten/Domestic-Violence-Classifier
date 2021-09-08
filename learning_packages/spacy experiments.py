import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp("I don't like this Mr. Bin")
for token in doc:
    print(token.text, token.pos_, token.dep_, token.lemma_)
print(len(doc))
print(len(doc.vocab))
doc2 = nlp(u'I am 7 weeks no contact and I left him 9 weeks ago. This is the longest I have gone no contact. I’ve finally broken the trauma bond and I am so proud of myself. I am feeling so free and so happy . He put me through hell for 2 and a half years. I finally know my worth and I will never look back, only forward.')
for sentence in doc2.sents:
    print(sentence)
print(len(doc2.vocab))

doc3 = nlp(u'New York is a beautiful city')
for chunk in doc3.noun_chunks:
    print(chunk)

doc4 =nlp("Hello there! Welcome to the programming world.")
for token in doc4:
    print(token.text)

#lemmatization
#Stemming just removes or stems the last few characters of a word, often leading to incorrect meanings and spelling. Lemmatization considers the context and converts the word to its meaningful base form, which is called Lemm
doc5= nlp("this runner is running 5 km.")
for token in doc5:
    print(token.lemma_)

#remove stop words
#print all stop words
stop_words = nlp.Defaults.stop_words
#add stop word
#nlp.nlp.Defaults.stop_words.add('btw')

#remove stop words:
#nlp.nlp.Defaults.stop_words.remove('btw')

filtered_sentence =[]

for word in doc5:
    lexeme = nlp.vocab[word.text]
    if lexeme.is_stop == False and lexeme.is_punct == False:
        filtered_sentence.append(word)

print(filtered_sentence)
#is punct- for punctuatuion