import os
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
import sklearn # use version 19.2
from nltk.corpus import stopwords
import nltk
from datasketch.minhash import MinHash
from datasketch.weighted_minhash import WeightedMinHashGenerator
from datasketch.lsh import MinHashLSH
import tqdm
import re
from nltk.tokenize.toktok import ToktokTokenizer
import datasketch as ds 
import time

def create_minHash_signatures(set_dict, no_of_permutations):

    """
    no_of_permutations is the number of permutations we want for the MinHash algorithm (discussed before). The higher the permutations the longer the runtime.
    Min_dict maps question id (eg 'm23') to min hash signatures.
    We loop through all the set representations of questions and calculate the signatures and store them in the min_dict dictionary.
    """

    min_dict = {}
    count = 0
    for val in set_dict.values():
       m = MinHash(num_perm=no_of_permutations)
       for shingle in val:
           m.update(shingle.encode('utf8'))
       min_dict["m{}".format(count)] = m
       count+=1
    return min_dict
    
def calculate_similarities(dictionary ,lsh):
    big_list = []
    number_of_duplicates=0
    duplicates=[]
    for i ,  query in enumerate(dictionary.keys()):
        #print(i)
        bucket = lsh.query(dictionary[query])
        duplicates.append((i,bucket))
        #print(str(i))
        #print(bucket)
        #number_of_duplicates+=len(bucket)
        if (len(bucket)>0):
            number_of_duplicates+=1
    

    return duplicates , number_of_duplicates
  
def get_shingles(text, char_ngram):

    return set(text[head:head + char_ngram] for head in range(0, len(text) - char_ngram))
    
import nltk
import re

def find_number_of_duplicates_between_sets(train,test,threshold,char_ngram, no_of_permutations):

    shingles_train = []
    set_dict={}
    norm_dict={} 
    for index, question in enumerate(train):

        shingles_train.append(get_shingles(question.lower(),char_ngram))
        set_dict["m{0}".format(index)]=get_shingles(question.lower(),char_ngram)
        norm_dict["m{0}".format(index)] = question

    shingles_test = []    
    test_set_dict={}
    test_norm_dict={}    
    for index, question in enumerate(test):

        shingles_test.append(get_shingles(question.lower(),char_ngram)) 
        test_set_dict["m{0}".format(index)]=get_shingles(question.lower() , char_ngram)
        test_norm_dict["m{0}".format(index)] = question
    
    #print(test_set_dict)
    #print(test_norm_dict)   
    #print (set_dict)
    #print (norm_dict)

    build_start_time = time.time()
    min_dict_test = create_minHash_signatures(test_set_dict, no_of_permutations)
    min_dict = create_minHash_signatures(set_dict, no_of_permutations)

    lsh = MinHashLSH(threshold, no_of_permutations)
    for key in min_dict.keys():
       lsh.insert(key,min_dict[key])
    build_time=time.time() - build_start_time
    query_start_time = time.time() 
    duplicates , number_of_duplicates =calculate_similarities(min_dict_test , lsh)
    import pandas as pd

    results=pd.DataFrame(duplicates, columns=['Test Doc. ID', 'Train Doc. ID'])
    #results.to_csv('Min_hash_'+str(char_ngram)+'_shingles_' + str(no_of_permutations) + '.csv', index=False, header=True)
    print(results.head(n=10))
    query_time=time.time() - query_start_time
    print("")
    print("Min-Hash LSH family. Set number of permutations to " +str(no_of_permutations))
    print("-------------------------------------")
    print("Number of duplicates = "+str(number_of_duplicates))
    print("Number of questions of test set that already exist in train = "+str(number_of_duplicates))
    print("Build time")
    print("--- %s seconds ---" % (build_time))
    print("Query time")
    print("--- %s seconds ---" % (query_time))
 

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

threshold=0.8
no_of_permutations = [ 16 , 32 ,64 ]
char_ngram = [3 , 4 ,5]
"""
data = [X_train[67859] , X_train[2470] , X_train[2457] ,X_train[328600], X_train[313502], X_train[149566], X_train[355437], X_train[5361], X_train[12]]
test = [X_test[1287] , X_test[5], X_test[5089], X_test[5087], X_test[51] , X_test[7]]
"""
for ngram in char_ngram:
    print("")
    print("*********************")
    print("Run for "+ str(ngram) + "-shingles")
    print("*********************")
    for number in no_of_permutations:

        print("")
        print("Number of permutations : " + str(number))
        print("---------------------------")
        print("")
        find_number_of_duplicates_between_sets(X_train , X_test,threshold, ngram , number)
  