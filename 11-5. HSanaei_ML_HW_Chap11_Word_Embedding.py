'''
Python Machine Learning
Teacher: Dr Rahmani
Student: Hossein SANAEI ~حسین سنایی
Homework Chapter 10

Aras International Campus of University of Tehran
Fall 1400 (2021)
GitHub: https://github.com/HSanaei/MachineLearing.git

Chapter 11  Word Embedding

'''

import gensim.downloader as api

model = api.load("glove-twitter-25")


vector = model['computer']
print('Word computer is embedded into:\n', vector)

similar_words = model.most_similar("computer")
print('Top ten words most contextually relevant to computer:\n', similar_words)



doc_sample = ['i', 'love', 'reading', 'python', 'machine', 'learning', 'by', 'example']

import numpy as np
doc_vector = np.mean([model[word] for word in doc_sample], axis=0)
print('The document sample is embedded into:\n', doc_vector)
