

import random
import logging
from numpy import array
from numpy import cumsum
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler
from pandas_ml import ConfusionMatrix
import matplotlib.pyplot as plt


# The BiLSTM
def Bi_LSTM(units, features, time_steps, prn=False):
    model = Sequential()
    model.add(Bidirectional(LSTM(units, return_sequences=False),
                            input_shape=(time_steps, features),
                            merge_mode='concat'))
    model.add(Dense(int((units/2)), activation='relu'))
    model.add(Dense(43, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['acc'])
    if prn:
        print(model.summary())
    return model


# Training 
# setting the task ; 3D modelling : sample, time steps, and feature.
model = Bi_LSTM(512, 400, None, prn=True)
# train model on given % of stimuli
# indicate the newtwork corpus:
# touples of shape (expected output(integer), sentence(series of vectors))
ts = int((len(network_corpus)*65)/100)
for i in range(ts):
    print('interation '+str(i)+'of'+' '+str(ts))
    x = [vec for vec in network_corpus[i][1]]
    y = network_corpus[i][0]
    # extracting each sample (sentence) lenth
    length = len(x)
    # reshape: sample, time steps, feature at each time step.
    # if I have 1000 sentences of 10 words, presented in a 3-dim vector:
    # is nb_samples = 1000, time steps =  10, input_dim = 3
    X = array(x).reshape(1, length, 400)
    Y = array(y).reshape(1, 1)
    model.fit(X, Y, epochs=10, batch_size=33, verbose=2)

# Evaluation
start = len(network_corpus)-ts
tt = 25
out = list(np.zeros(tt))
exp = list(np.zeros(tt))
correct = 0
for i in range(tt):
    x = [vec for vec in network_corpus[i+start][1]]
    y = network_corpus[i+start][0]
    length = len(x)
    X = array(x).reshape(1, length, 400)
    Y = array(y).reshape(1, 1)
    out[i] = model.predict_classes(X, verbose=2)
    exp[i] = y
    if y == out[i]:
        correct += 1
    print('predicted Class: '+str(out[i])+' Actual Class: '+
          str(y))
print('Overall accuracy: '+str(int((correct*100)/tt))+'%')

# plotting the confusion matrix
# reconverting numbers to presidnts names
predicted = [list(d.keys())[int(p)] for p in nltk.word_tokenize(str(out)) if p.isdigit()]
actual = [list(d.keys())[a] for a in exp]
cf = ConfusionMatrix(actual, predicted)
cf.plot(normalized=True, backend='seaborn', cmap="Blues")
plt.show()




