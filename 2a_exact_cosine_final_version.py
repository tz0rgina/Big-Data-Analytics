import os
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
import sklearn # use version 19.2
from nltk.corpus import stopwords
import nltk

from sklearn.feature_extraction.text import CountVectorizer
import tqdm
import re
import scipy
from nltk.tokenize.toktok import ToktokTokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import time
import sparse, numpy as np
from sklearn.metrics.pairwise import cosine_similarity 


oov_tok = '<OOV>' #OOV = Out of Vocabulary
trunc_type = 'post'
padding_type = 'post'

# TRAIN DATA
df_train=pd.read_csv("corpusTrain.csv")
#print(df_train.head())
df_train.loc[0]['Content']

X_train = df_train.iloc[:, 1].values
# print(X_train)
print("number of X_train dimensions: ", X_train.ndim)
print("number of X_train examples: ", len(X_train))

# TEST DATA
df_test=pd.read_csv("corpusTest.csv")
df_test.head()
df_test.loc[0]['Content']

X_test = df_test.iloc[:, 1].values
#print(X_test[:30])
print("number of X_test dimensions: ", X_test.ndim)
print("number of X_test examples: ", len(X_test))
#data = [X_train[247]]
data = [X_train[2470] , X_train[2457] , X_train[67859] ,X_train[328600], X_train[313502], X_train[149566], X_train[355437], X_train[5361], X_train[12]]
test = [X_test[1287] , X_test[5], X_test[5089], X_test[5087], X_test[51] , X_test[7]]

vectorizer=CountVectorizer(lowercase=True)
vectorizer.fit(X_train)

start_time=time.time()
train_vectorized = vectorizer.transform(data)

test_vectorized = vectorizer.transform(test) 
   
duplicates=[]
for i , train_doc  in enumerate(train_vectorized):
    #print(i)
    cosine_similarity_matrix = cosine_similarity(test_vectorized, train_doc)
    temp_similar =[list(np.where(cosine_similarity_matrix>=0.8))[0]] # list of similar questions in test corpus with train_doc
    #print(cosine_similarity_matrix)
    if(len(temp_similar[0])!=0):
        duplicates.extend([(j,i) for j in temp_similar[0]]) # Extend duplicates with pairs (test_data , train_data) with
#print(duplicates)                                           # cosine similarity larger than 0.8 
exact = len(np.unique([list(i)[0] for i in duplicates]))

#print(exact)

print("total (build+query) time is: ")
print("--- %s seconds ---" % (time.time() - start_time))
print("Number of questions of test set that already exist in train = "+str(exact)) 

results=pd.DataFrame(duplicates, columns=['Test Doc. ID', 'Train Doc. ID'])

print(pd.DataFrame(duplicates, columns=['Test Doc. ID', 'Train Doc. ID']))
results.to_csv('Exact_cosine_test.csv', index=False, header=True)



