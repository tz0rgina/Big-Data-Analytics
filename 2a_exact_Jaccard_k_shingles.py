import os
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
import sklearn # use version 19.2
import re
from nltk.tokenize.toktok import ToktokTokenizer 
import pandas as pd
import time

# TRAIN DATA
df_train=pd.read_csv("corpusTrain.csv" )
#print(df_train.head())
df_train.loc[0]['Content']

X_train = df_train.iloc[:, 1].values
# print(X_train)
print("number of X_train dimensions: ", X_train.ndim)
print("number of X_train examples: ", len(X_train))

# TEST DATA
df_test=pd.read_csv("corpusTest.csv" )
df_test.head()
df_test.loc[0]['Content']

X_test = df_test.iloc[:, 1].values
#print(X_test[:30])
print("number of X_test dimensions: ", X_test.ndim)
print("number of X_test examples: ", len(X_test))

# a pure python shingling function that will be used in comparing
# LSH to true Jaccard similarities
def get_shingles(text, char_ngram):
    #print(set(text[head:head + char_ngram] for head in range(0, len(text) - char_ngram)))
    return set(text[head:head + char_ngram] for head in range(0, len(text) - char_ngram))

def jaccard(set_a, set_b):
    intersection = set_a & set_b
    union = set_a | set_b
    #print(intersection)
    #print(len(intersection))
    #print(union)
    #print(len(union))
    return len(intersection) / len(union)
 ]
def run(char_ngram):

    start_time = time.time()
    
    print("Run for " + str(char_ngram) + "-shingles.")
    print("----------------------")
    print("")
    
    shingles_train = []
    shingles_test = []
    
    for index, question in enumerate(X_train):
        shingles_train.append(get_shingles(question.lower() , char_ngram))
    
    for index, question in enumerate(X_test):
        shingles_test.append(get_shingles(question.lower() , char_ngram)) 

    print(shingles_test)
    print(shingles_train)

    duplicates = []
    exact=0
    
    for i_doc in range(len(shingles_test)):
        print(i_doc)
        #print(X_test[i_doc])
        has_duplicate=False
        if(len(shingles_test[i_doc])!=0): # Some questions in test corpus are empty
            for j_doc in range(len(shingles_train)):
                if(len(shingles_train[j_doc])!=0): # Some questions in train corpus are empty
                    jaccard_similarity = jaccard(shingles_test[i_doc], shingles_train[j_doc])
                    print(jaccard_similarity)
                    if (jaccard_similarity >= 0.8):
                        duplicates.append((i_doc, j_doc, jaccard_similarity))

    exact = len(np.unique([list(i)[0] for i in duplicates]))
    #print(exact)
         
    #print train time
    print("total time is: ")
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Number of questions of test set that already exist in train = "+str(exact))

    results=pd.DataFrame(duplicates, columns=['Test Doc. ID', 'Train Doc. ID', 'Jaccard Similarity'])
    print(pd.DataFrame(duplicates, columns=['Test Doc. ID', 'Train Doc. ID', 'Jaccard Similarity']).head(n=10))
    results.to_csv('Exact_Jaccard_' + str(char_ngram) + '_shingles.csv', index=False, header=True)

    
char_ngram = [3 , 4 , 5] 

for ngram in char_ngram:
    
    run(ngram)
