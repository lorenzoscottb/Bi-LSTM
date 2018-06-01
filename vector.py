
import os
import re
import nltk
import random
import logging
import numpy as np
import string
from glob import glob
from numpy import array
from nltk import WordNetLemmatizer 
from nltk.corpus import wordnet
from gensim.models import Word2Vec
from nltk.corpus import stopwords


# Stop words
en_stop = stopwords.words('english')
en_stop.append("'d")
it_stop = stopwords.words('italian')
punctuations = list(string.punctuation)

# Functions 
def clean_sentences(sent_list):

    def lemmatizer(toupla):

        lm = WordNetLemmatizer()

        if toupla[1].startswith('J'):
            return lm.lemmatize((toupla[0]), wordnet.ADJ)
        elif toupla[1].startswith('V'):
            return lm.lemmatize((toupla[0]), wordnet.VERB)
        elif toupla[1].startswith('N'):
            return lm.lemmatize((toupla[0]), wordnet.NOUN)
        elif toupla[1].startswith('R'):
            return lm.lemmatize((toupla[0]), wordnet.ADV)
        else:
            return lm.lemmatize(toupla[0])

    # word tokenization
    sents = [nltk.word_tokenize(sent) for sent in sent_list]

    # remove digits
    print('removing digits')
    for s in range(len(sents)):
        for tk in range(len(sents[s])):
            if sents[s][tk].isdigit():
                sents[s][tk] = '#cardinal'

    # removing stopwords and punctuation
     print('removing stop words and punctuation')
    clean_sents = list(np.zeros(len(sents)))
    for i in range(len(sents)):
        clean_sents[i] = [word.lower() for word in sents[i] if word.lower()
                          not in en_stop and word.lower() not in punctuations]

    # pos tagging
    print('pos tagging')
    tag_sent = [nltk.pos_tag(sent) for sent in clean_sents]

    # lemmatizing (still needs to use the pos tag)
     print('lemmatizing')
    final_sent = list(np.zeros(len(tag_sent)))
    for s in range(len(tag_sent)):
        final_sent[s] = [lemmatizer(touple) for touple in tag_sent[s]]

    return final_sent


# Preparing the corpus
print('Input directory to Word2Vec trining corpora')

corpus = input()

print('creating the corpus')
pg = []
documents = []
# Needs a folder filled with texts to read
for file in glob(corpus + os.sep + '**', recursive=True):
    if os.path.isdir(file):
        continue
    doc = open(file, 'r').read()
    documents.append(doc)
    s = nltk.sent_tokenize(doc)
    for sent in s:
        pg.append(str(sent).strip('[', ).strip(']'))

print('the corpus is done', '\nit contains', len(documents),
      'documents and', len(pg), 'sentences')

pirnt('Cleaning the collected Corpus')
final_corpus = clean_sentences(pg)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

# sentences vectorization
# set the 'new' method
languageModel = Word2Vec(fianl_corpus, size=400, min_count=0)



