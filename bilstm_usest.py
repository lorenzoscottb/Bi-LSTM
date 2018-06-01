

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

# plotting model's confusion matrix
cf = ConfusionMatrix(y, prediction)
cf.plot()
plt.show()







