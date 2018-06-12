"""
Uses the U.S. preidential speeches corpus to create a Presidens x Year x Sentece Corpus (already downloadable in repository)
and the Pres(2num) X vector (sentences) corpus
The code is an extention of the work from Patrizio Bellan (patriziobellan86)

"""



import os
import re
import nltk
import random
import logging
import numpy as np
import string
from glob import glob
from numpy import array
from numpy import cumsum
from nltk import WordNetLemmatizer 
from nltk.corpus import wordnet


# Stop words
en_stop = stopwords.words('english')
en_stop.append("'d")
it_stop = stopwords.words('italian')
punctuations = list(string.punctuation)

# Functions

def hgs_len(list_of_sent):

    """""""""
    returns the longest sent
    """

    ln = [len(s) for s in list_of_sent]
    m = max(set(ln))

    return m


def max_len(list_of_sent):

    ln = [len(s) for s in list_of_sent]
    m = max(set(ln), key=ln.count)

    return m, ln.count(m)

def clean_sentences(sent_list):

    """""""""
    take a list of str and oprates statard corpus cleaning
    """""

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
    print('tokenizing sentences')
    sn = list(np.zeros(len(sent_list)))
    for sen in range(len(sn)):
        sn[sen] = nltk.word_tokenize(sent_list[sen])

    # remove digits
    print('removing digits')
    for s in range(len(sn)):
        for tk in range(len(sn[s])):
            if sn[s][tk].isdigit():
                sn[s][tk] = '#cardinal'

    # removing stopwords and punctuation
    print('removing stop words and punctuation')
    clean_sents = list(np.zeros(len(sn)))
    for i in range(len(sn)):
        clean_sents[i] = [word.lower() for word in sn[i] if word.lower()
                          not in en_stop and word.lower() not in punctuations]

    # pos tagging
    print('pos tagging')
    tag_sent = list(np.zeros(len(clean_sents)))
    for i in range(len(clean_sents)):
        tag_sent[i] = nltk.pos_tag(clean_sents[i])

    # lemmatizing (still needs to use the pos tag)
    print('lemmatizing')
    final_sent = list(np.zeros(len(tag_sent)))
    for s in range(len(tag_sent)):
        final_sent[s] = [lemmatizer(touple) for touple in tag_sent[s]]

    return final_sent


# input the directory for corpora
print('set directory for U.S. Speaches Corpora)
folder = input() 

# writing pres_X_year_X_sent corpus N.B.: not cleaned
speeches = []
for file in glob(folder + os.sep + '**', recursive=True):
    if os.path.isdir(file):
        # if it is a folder continue
        continue
    # read Speech file
    corpus = open(file, 'r').readlines()

    # extracting President Name
    presidentName = os.path.basename(file)[:os.path.basename(file).find('_')]
    speechYear = corpus[1][corpus[1].find(',') + 1:corpus[1].rfind('"')].strip()

    # cleanning annotation out from the speech
    speech = corpus[2]

    speech = speech.replace('\n', ' ')

    # regular expression for cleanning
    pattern = re.compile('<.*>')
    speech = re.sub(pattern, '', corpus[2])

    speech = speech.strip()

    sentences = nltk.sent_tokenize(speech)

    # write corpus into file. PresidentName, data,
    #  speech_sentence for each line
    f = open(folder + os.sep + 'Corpus', 'a')
    for line in sentences:
        f.write(presidentName + ' ____ ' + speechYear + ' ____ ' + line + '\n')
    f.close()

datasetType = ['name', 'year']  # 1 = year, 0 = name

# directory for the splitted corpus
print('set directory for the splitted corupus)
corpus = input()

# test on  name
datasetType = datasetType[0]

# read corpus
corpus = open(corpus, 'r').readlines()

# shuffling samples
random.shuffle(corpus)

# transformation into tuple (presidentName, year, sentence)
corpus = [c.strip().split(' ____ ') for c in corpus]

# sentences extraction (now 80 list have no sentence)
sent = [sent[2] for sent in corpus if len(sent) == 3]
clean_sents = clean_sentences(sent)

#  Possible presidents and pres2number table of conversion
if datasetType == 'name':
    indice = 0
else:
    indice = 1
outputs = list(set([c[indice] for c in corpus]))
conv_table = [(outputs[i], i) for i in range(len(outputs))]
d = dict(conv_table)
datiOutput = [c[indice] for c in corpus]
pres2num = [d[pres] for pres in datiOutput]
# in case you need binary hot-vectors
pres2categorical = to_categorical(pres2num, num_classes=43)


# Creating the corpus for the network
network_corpus = []
for i in range(len(clean_sents)):
    csent = clean_sents[i]
    if len(csent) == 0:
        continue
    vectors = [languageModel.wv[word] for word in csent]
    network_corpus.append((pres2num[i], vectors))
