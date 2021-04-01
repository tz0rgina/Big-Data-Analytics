import os
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
import sklearn # use version 19.2
import re
from nltk.tokenize.toktok import ToktokTokenizer 
import pandas as pd

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

def jaccard(set_a, set_b):
    intersection = set_a & set_b
    union = set_a | set_b
    print (union)
    print (len(union))
    print (intersection)
    print (len(intersection))
    return len(intersection) / len(union)
        
def remove_special_characters(text):
        pattern=r'[^a-zA-z0-9\s]'
        return(re.sub(pattern,'',text))
 
def set_representation(data):
    shingles=[]
    for i,question in enumerate(data):
        #print(question)
        word_list = []
        question=remove_special_characters(question)
        tokenizer=ToktokTokenizer()
        words = tokenizer.tokenize(question)
        for word in words:
            word_list.append(word.lower())
        #print(set(word_list))
        shingles.append(set(word_list))
        
    return shingles
    
"""  
Examples which we used for debugging and as examples for the report
data = [X_train[67859] , X_train[2470] , X_train[2457] ,X_train[328600], X_train[313502], X_train[149566], X_train[355437], X_train[5361], X_train[12]]
test = [X_test[1287] , X_test[5], X_test[5089], X_test[5087], X_test[51] , X_test[7]]
""" 
 
import time
start_time = time.time()

shingles_train=set_representation(X_train)
shingles_test=set_representation(X_test)

duplicates = []
exact=0
for i_doc in range(len(shingles_test)):
    print(i_doc)
    #print(X_test[i_doc])
    has_duplicate=False
    if(len(shingles_test[i_doc])!=0):
        for j_doc in range(len(shingles_train)):
            #print("")
            #print(test[i_doc])
            #print(data[j_doc])
            if(len(shingles_train[j_doc])!=0):
                jaccard_similarity = jaccard(shingles_test[i_doc], shingles_train[j_doc])
                #print(jaccard_similarity)
                is_duplicate = jaccard_similarity >= 0.8
                if is_duplicate:
                    duplicates.append((i_doc, j_doc, jaccard_similarity))


exact = len(np.unique([list(i)[0] for i in duplicates]))
#print(exact)
     
#print train time
print("total (build+query) time is: ")
print("--- %s seconds ---" % (time.time() - start_time))
print("Number of questions of test set that already exist in train = "+str(exact))

results=pd.DataFrame(duplicates, columns=['Test Doc. ID', 'Train Doc. ID', 'Cosine Similarity'])
print(pd.DataFrame(duplicates, columns=['Test Doc. ID', 'Train Doc. ID', 'Cosine Similarity']).head(n=10))
results.to_csv('Exact_Jaccard_word_shingles.csv', index=False, header=True)