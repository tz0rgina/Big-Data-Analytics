import numpy as np
import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn 
import collections
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity 
import time


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
#df_test=pd.read_csv("corpusTest.csv")
df_test.head()
df_test.loc[0]['Content']

X_test = df_test.iloc[:, 1].values


print("number of X_test dimensions: ", X_test.ndim)
print("number of X_test examples: ", len(X_test))
tfidf = TfidfVectorizer(lowercase=True,
                        max_features=10000)
"""
data = [ X_train[67859] , X_train[2470] , X_train[2457]  ,X_train[328600], X_train[313502], X_train[149566], X_train[355437], X_train[5361], X_train[12] , X_test[5089]]
test = [X_train[67859] ,X_test[1287] , X_test[5], X_test[5089], X_test[5087], X_test[51] ,  X_test[7]]
"""
# TFIDF for train data
X= tfidf.fit_transform(X_train)
# TFIDF for test data
Xt = tfidf.transform(X_test) 
#X = tfidf.transform(data) 
    
class HashTable:
    def __init__(self, hash_size, inp_dimensions):
        self.hash_size = hash_size
        self.hash_table = dict()
        self.projections = np.random.randn(self.hash_size, inp_dimensions)
        #print(self.projections)

    def generate_hash(self, inp_vector):
        bools = (np.dot(inp_vector, self.projections.T) > 0).astype('int')
        return ''.join(str(v) for v in bools)
    
    def __setitem__(self, inp_vec, label):
        hash_value = self.generate_hash(inp_vec)
        self.hash_table[hash_value] = self.hash_table\
            .get(hash_value, list()) + [label]
        

    def __getitem__(self, inp_vec):
        hash_value = self.generate_hash(inp_vec)
        #print(self.hash_table)
        result=[]
        if hash_value in self.hash_table:
            #print(hash_value)
            #print(self.hash_table[hash_value])
            result=self.hash_table[hash_value]
        return result


class LSH:
    def __init__(self, num_tables, hash_size, inp_dimensions):
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.inp_dimensions = inp_dimensions
        self.hash_tables = list()
        for i in range(self.num_tables):
            self.hash_tables.append(HashTable(self.hash_size, self.inp_dimensions))
    
    def __setitem__(self, inp_vec, label):
        for table in self.hash_tables:
            table[inp_vec] = label
    
    def __getitem__(self, inp_vec):
        results = list()
        for table in self.hash_tables:
            #print(table)
            #print(table[inp_vec])
            if(len(table[inp_vec])!=0):
                results.extend(table[inp_vec])
                #print(results)
        return results
        
    def describe(self):
        for table in self.hash_tables:
            print (table.hash_table)

class sameQuestionSearch:

    def __init__(self, num_tables, hash_size, inp_dimensions , training_files): 
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.inp_dimensions = inp_dimensions
        self.lsh = LSH(self.num_tables, self. hash_size, self.inp_dimensions)
        self.training_files = training_files
            
    def train(self):
        for i , x in enumerate(self.training_files):
            self.lsh[x.toarray()]= str(i)  

        
    
    def find_similar_question(self, question , candidates_array): 
        #print(question)
        frequency=[]                                     
        similar=[]

        candidates = np.unique(candidates_array)
        candidates = [int(c) for c in candidates]
        #print(candidates)
       
        cosine_similarity_matrix = cosine_similarity(question, X[candidates])
        #print(cosine_similarity_matrix)
        temp_similar =[list(np.where(cosine_similarity_matrix[0]>=0.8))[0]] # list of similar questions in test corpus with 
        #print(list(temp_similar[0]))
        #print("---")
        if(len(temp_similar[0])!=0):                                 # cosine similarity larger than 0.8 
            similar=[candidates[i] for i in list(temp_similar[0])]
        #print(similar)
        return similar


        
    def query(self, questions):
        exact=0
        multiple=0
        duplicates =[]
        for i, question in enumerate(questions):
            res = self.lsh[question.toarray()]
            #print(test[i])
            #print(res)
            if (len(res)>0):
                
                similar = self.find_similar_question(question, res)
                if (len(similar)>0):
                    exact += 1
                    multiple += len(similar)
                duplicates.append((i, similar))
            

        print("Number of duplicates = "+str(multiple))
        print("Number of questions of test set that already exist in train = "+str(exact))

        df_results=pd.DataFrame(duplicates, columns=['Test Doc. ID', 'Train Doc. ID'])
        print(pd.DataFrame(duplicates, columns=['Test Doc. ID', 'Train Doc. ID']).head(n=10))
        df_results.to_csv('Cos_LSH_k'+ str(self.hash_size)+ '_L' + str(self.num_tables) +'.csv', index=False, header=True)
 

k=[1, 2, 3, 4 , 5, 6 , 7,  8 , 9, 10, 17]
L=[1, 1, 1, 1, 1, 1 , 1 , 1 , 1  ,1 , 40]

for hash_size , num_tables   in zip(k,L):
    print("")
    print("LSH with cosine - " + str(hash_size) + " random hyperplanes")
    print("")
    inp_dimensions = 10000
    search= sameQuestionSearch(num_tables, hash_size+1, inp_dimensions , X) 
    build_start_time = time.time()  
    search.train()
    build_time=time.time() - build_start_time
    query_start_time = time.time() 
    search.query(Xt)
    query_time=time.time() - query_start_time
    print("Build time")
    print("--- %s seconds ---" % (build_time))
    print("Query time")
    print("--- %s seconds ---" % (query_time))
