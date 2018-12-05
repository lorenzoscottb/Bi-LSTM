
# Bi-LSTM for (semantic's) author-recognition (?)

Author profiling algorithms are indeed now-days efficient, but freqeuntly rely on (very) large sets of features. with this
task  I tried to investigate (as work for my intro to programming class) if a similar classification problem,
that identifying the author of a speech, using (word2vec) embedding.
 To do so I implemented a Bilateral LSMTM. The classification is on 43 united stated elected presidents 
 and is done based on their discourses. 

### Requirments

- nltk
- random
- logging
- numpy 
- gensim
- keras
- sklearn
- pandas_ml 
- matplotlib
            
The contained files are: the network script, the zip of the U.S. presidential speeches as well as the already prepared 
President - Year - sentence corpus.
This last two were scripted by Patrizio Bellan (patriziobellan86) for a previous project. 
