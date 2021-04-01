import os
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
import sklearn # use version 19.2\
from sklearn.feature_extraction.text import TfidfVectorizer
import cleanData
from sklearn.feature_extraction.text import CountVectorizer
import time
import pandas as pd

def cosine(vector1 , vector2):
    print(np.dot(vector1 , vector2))
    if(np.linalg.norm(vector1)==0 or np.linalg.norm(vector2)==0):
        cosine = 0.0
    else:
        print(np.linalg.norm(vector1)*np.linalg.norm(vector2))
        cosine = np.dot(vector1 , vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))
    return cosine
    


  def cosine(vector1 , vector2):
    #print(vector1)
    #print(vector2)
    norm1=np.sum(np.abs(vector1)**2,axis=-1)**(1./2)
    norm2=np.sum(np.abs(vector2)**2,axis=-1)**(1./2)
    norm1 = norm1.reshape(len(norm1),1)
    cosine = np.dot(vector1 , np.transpose(vector2))/(norm1*norm2)
    return cosine
    
# TRAIN DATA
df_train=pd.read_csv("corpusTrain.csv")
df_train.loc[0]['Content']

X_train = df_train.iloc[:, 1].values
print("vectorize -> BoW approach")
vectorizer=CountVectorizer(max_features=10000 , lowercase=True)

#Examples which we used for debugging and as examples for the report
data = [X_train[67859] , X_train[2470] , X_train[2457] ,X_train[328600], X_train[313502], X_train[149566], X_train[355437], X_train[5361], X_train[12]]
test = [X_test[1287] , X_test[5], X_test[5089], X_test[5087], X_test[51] , X_test[7]]

# Vectorize train data
vectorizer.fit(X_train)
X_train = vectorizer.transform(data)

print("Test data processing")
# TEST DATA
df_test=pd.read_csv("corpusTest.csv")
df_test.head()
df_test.loc[0]['Content']

X_test = df_test.iloc[:, 1].values

print("number of X_test dimensions: ", X_test.ndim)
print("number of X_test examples: ", len(X_test))

 
# Vectorize test data
start_time = time.time()
X_test = vectorizer.transform(test)
#print(test)
X_test = X_test.toarray()	
#print(X_train)

#print(X_test.shape)
#print(X_train.shape)
duplicates = []
exact=0
for i_doc in range(X_test.shape[0]):
    print(i_doc)
    #test_doc = X_test[j_doc].toarray()
    for j_doc in range(X_train.shape[0]):
        #print("")
        #print(i_doc)
        #print(j_doc)
        train_doc = X_train[j_doc].toarray()
        print(train_doc[0])
        print(X_test[i_doc])
        cosine_similarity = cosine(X_test[i_doc], train_doc[0])
        
        #print(cosine_similarity)
        #print("")
        is_duplicate = cosine_similarity >= 0.8
        if is_duplicate:
            duplicates.append((i_doc, j_doc, cosine_similarity))
            

exact = len(np.unique([list(i)[0] for i in duplicates])) 
 
#print train time
print("total (build+query) time is: ")
print("--- %s seconds ---" % (time.time() - start_time))
print("Number of duplicates = "+str(len(duplicates)))
print("Number of questions of test set that already exist in train = "+str(exact))

results=pd.DataFrame(duplicates, columns=['Test Doc. ID', 'Train Doc. ID', 'Cosine Similarity'])
print(pd.DataFrame(duplicates, columns=['Test Doc. ID', 'Train Doc. ID', 'Cosine Similarity']).head(n=10))
#results.to_csv('Exact_cosine.csv', index=False, header=True)
