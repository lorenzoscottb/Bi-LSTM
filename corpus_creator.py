"""
Uses the U.S. preidential speeches corpus to create a Presidens x Year x Sentece Corpus (already downloadable in repository)
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
def max_len(list_of_sent):

    ln = [len(s) for s in list_of_sent]
    m = max(set(ln), key=ln.count)

    return m, ln.count(m)

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

    # rename digits
    for s in range(len(sents)):
        for tk in range(len(sents[s])):
            if sents[s][tk].isdigit():
                sents[s][tk] = '#cardinal'

    # removing stopwords and punctuation
    clean_sents = list(np.zeros(len(sents)))
    for i in range(len(sents)):
        clean_sents[i] = [word.lower() for word in sents[i] if word.lower()
                          not in en_stop and word.lower() not in punctuations]

    # pos tagging
    tag_sent = [nltk.pos_tag(sent) for sent in clean_sents]

    # lemmatizing (still needs to use the pos tag)
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


# Creating the matched corpus for train and test
# problem: there are 35 classes, solution: searche for len od sent
# that allow to have example from all pres
length, data_length = max_len(clean_sents)
network_corpus = []
for i in range(len(clean_sents)):
    if len(clean_sents[i]) == length:
        vectors = [languageModel.wv[word] for word in clean_sents[i]]
        network_corpus.append((pres2num[i], vectors))

# # normalize vectors
# scaler = StandardScaler()
# normalize_vec = scaler.transform(vectors)
# for i in range(data_length):
#     vec = []
#         for v in range(i, i+11):
#            vec.append()
#     network_corpus.append((pres2num[i], vectors[i]))
