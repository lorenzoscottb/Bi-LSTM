




# Stop words
en_stop = stopwords.words('english')
it_stop = stopwords.words('italian')
punctuations = list(string.punctuation)


def max_len(list_of_sent):

    ln = [len(s) for s in list_of_sent]
    m = max(set(ln), key=ln.count)

    return m, ln.count(m)


def clean_sentences(sent_list):

    # word tokenization
    sents = [nltk.word_tokenize(sent) for sent in sent_list]

    # remove digits
    for s in range(len(sents)):
        for tk in range(len(sents[s])):
            if sents[s][tk].isdigit():
                sents[s][tk] = '#cardinal'

    # removing stopwords and punctuation
    clean_sents = list(np.zeros(len(sents)))
    for i in range(len(sents)):
        clean_sents[i] = [word for word in sents[i] if word.lower()
                          not in en_stop and word.lower() not in punctuations]

    # pos tagging
    tag_sent = [nltk.pos_tag(sent) for sent in clean_sents]

    # lemmatizing (still needs to use the pos tag)
    lemmatizer = WordNetLemmatizer()
    final_sent = list(np.zeros(len(tag_sent)))
    for s in range(len(tag_sent)):
        final_sent[s] = [lemmatizer.lemmatize(touple[0]) for
                         touple in tag_sent[s]]

    return final_sent


usps = '/Users/lorenzoscottb/Documents/corpora/usps'
ct = '/Users/lorenzoscottb/Documents/corpora/clintonvstrump'

folder = usps
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
cupsp = '/Users/lorenzoscottb/Documents/corpora/usps/Corpus'
cct = '/Users/lorenzoscottb/Documents/corpora/clintonvstrump/Corpus1'

corpus = cupsp

# test on  name
datasetType = datasetType[0]

# read corpus
corpus = open(corpus, 'r').readlines()

# shuffling samples
random.shuffle(corpus)

# transformation into tuple (presidentName, year, sentence)
corpus = [c.strip().split(' ____ ') for c in corpus]

# sentences extraction (now 80 list have no sentence)
# sentences = [sent[2] for sent in corpus]
sent = [sent[2] for sent in corpus if len(sent) == 3]
clean_sents = clean_sentences(sent)


# for info from model
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

# sentences vectorization
# set the 'new' method
languageModel = Word2Vec(clean_sents, size=400, min_count=0)


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


# The BiLSTM
def Bi_LSTM(units, features, time_steps, prn=False):
    # setting the task ; 3D modelling : sample, time steps, and feature.
    model = Sequential()
    # input shape: (given units, how many time steps)
    # Bidirectional RNN can concatenate, using merge_mode='concat' !!!
    # input_shape = (time_steps, features)
    model.add(Bidirectional(LSTM(units, return_sequences=False),
                            input_shape=(time_steps, features),
                            merge_mode='concat'))
#    model.add(Dense(int((units/2)), activation='relu'))
    model.add(Dense(43, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['acc'])
    if prn:
        print(model.summary())
    return model


# Training
model = Bi_LSTM(512, 400, length, prn=True)
# train model on given % of stimuli
ts = int((data_length*65)/100)
x = list(np.zeros(ts))
y = list(np.zeros(ts))
for i in range(ts):
    x[i] = [vec for vec in network_corpus[i][1]]
    y[i] = network_corpus[i][0]
# reshape: sample, time steps, feature at each time step.
# if I have 1000 sentences of 10 words, presented in a 3-dim vector:
# is nb_samples = 1000, time steps =  10, input_dim = 3
X = array(x).reshape(ts, length, 400)
Y = array(y).reshape(ts, 1)
model.fit(X, Y, epochs=50, batch_size=33, verbose=2)


# Evaluation
tt = data_length-ts
x = list(np.zeros(tt))
y = list(np.zeros(tt))
for i in range(tt):
    x[i] = [vec for vec in network_corpus[i+ts][1]]
    y[i] = network_corpus[i+ts][0]

X = array(x).reshape(tt, length, 400)
Y = array(y).reshape(tt, 1)
yhat = model.predict_classes(X, verbose=2)
correct = 0
for i in range(tt):
    exp = y[i]
    pred = yhat[i]
    if exp == pred:
        correct += 1
    print('predicted Class: '+str(yhat[i])+' Actual Class: '+
          str(y[i])+'')
prediction = list(yhat)
print('Overall accuracy: '+str(int((correct*100)/tt)))
# resetting labels


# plotting confusion matrix
cf = ConfusionMatrix(y, prediction)
cf.plot()
plt.show()







