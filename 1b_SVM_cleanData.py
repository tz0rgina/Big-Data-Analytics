from scipy.sparse import random as sparse_random
from sklearn.random_projection import sparse_random_matrix

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from numpy import mean
from numpy import std
import numpy as np

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import make_scorer
import seaborn as sns
import matplotlib.pyplot as plt

import cleanData

#calculate execution time
import time


#import pandas
import pandas as pd
print("imported pandas...")

def clean_data(data):
    for i, text in enumerate(data):
        txt=cleanData.DataCleaner(text)
        data[i]=txt.clean()
    return data

def get_models():

    models = dict()
    models["SVM (BoW)"]=SGDClassifier(loss='hinge' , max_iter = 500, penalty='elasticnet')

    return models

#Training the model on full traiing set 
#Predictions on test set and printing classificatio report
#Predictions on ublabeled data and save to csv file
def report_and_predictions(model_name, model , X_train, y_train, X_test, y_test , X_test_unlabeled):

    start_time = time.time()

    model.fit(X_train, y_train)
    
    print("Time for training with all training set ") 
    print("--- %s seconds ---" % (time.time() - start_time))
    print("")

    print("Classification report for " + model_name) 
    print("---------------------------------------") 
 
    yhat = model.predict(X_test)
    
    print(classification_report(y_test, yhat))
    
    # make final KAGGLE predictions with classification
    print("now make KAGGLE predictions")
    yhat_pred_kaggle = model.predict(X_test_unlabeled)

    #EXPORT RESUTLS for KAGGLE
    print("export results to .csv")
    results = pd.DataFrame({'Id':df_test_final['Id'], 'Predicted':yhat_pred_kaggle})
    results.to_csv('cleaned_unlabeled_predictions_09_11_'+model_name+'.csv', index=False, header=True)
    
def crossValidation_and_evaluation(X, y , X_test, y_test, X_test_unlabeled):

    print(y)
    print(X.shape, y.shape)

    models = get_models()
    
    CV = 5
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    print(cv_df)
    entries = []
    means = []

    custom_scorer = {'accuracy':make_scorer(accuracy_score),
                     'precision': make_scorer(precision_score, average='macro'),
                     'recall': make_scorer(recall_score, average='macro'),
                     'f1': make_scorer(f1_score, average='macro'),
                    }

    for model_name, model in models.items():
        
        print("")
        print ("Cross Validation for " + model_name)
        print("")
        
        scores = cross_validate(model, X, y, scoring=custom_scorer , cv=CV)
        #print(scores)
        # Get the regular numpy array from the MaskedArray
        
        accuracies=np.array(scores['test_accuracy'])
        precisions= np.array(scores['test_precision'] )
        recalls = np.array(scores['test_recall'])
        f1s =np.array(scores['test_f1'])
        
        for fold_id in range(CV):
            entries.append((model_name, fold_id, accuracies[fold_id], precisions[fold_id] , recalls[fold_id],
                        f1s[fold_id]))
                        
        means.append([ accuracies.mean() , precisions.mean(),recalls.mean() ,f1s.mean() ])
        
        report_and_predictions(model_name, model , X, y, X_test, y_test, X_test_unlabeled)
        
    return means , entries

print("now read train.csv")
#read train data
df_train=pd.read_csv("./train.csv")
print(df_train.head())
y_train = df_train['Label']

df_train.Label.unique()
#df_train.loc[0]['Title']
#df_train.loc[0]['Content']
#df_train.loc[0]['Label']

print("now read test.csv")
#read test data
df_test=pd.read_csv("./test.csv")
print(df_test.head())
y_test = df_test['Label']


X_train = df_train.iloc[:, 1] + " " + df_train.iloc[:, 2] # HERE concatenate Title with Content
print(X_train[1])
X_test = df_test.iloc[:, 1] + " " + df_test.iloc[:, 2] # HERE concatenate Title with Content
#print(X_test[1])

print("now read test_without_labels.csv")
#read test data
df_test_final=pd.read_csv("test_without_labels.csv")
X_test_final = df_test_final.iloc[:, 1].values + " " + df_test_final.iloc[:, 2].values  # HERE concatenate Title with Content
#print(X_test_final[1])

print("Clean data")
#Clean Data from stopwords, special characters, digits and to lower case
X_train = np.array(clean_data(X_train))
print(X_train[1])
X_test = np.array(clean_data(X_test))
X_test_final = np.array(clean_data(X_test_final))
from sklearn.feature_extraction.text import TfidfVectorizer

print("vectorize with TF-IDF -> BoW approach")
tfidf = TfidfVectorizer(lowercase=True)

# TFIDF for train data
X_train = tfidf.fit_transform(X_train)

# TFIDF for test data
X_test = tfidf.transform(X_test)

# TFIDF for test data
X_test_unlabeled = tfidf.transform(X_test_final)

#5 fold Cross Validation for all models

means, entries = crossValidation_and_evaluation(X_train,y_train , X_test, y_test , X_test_unlabeled)
print (means)
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy', 'precision', 'recall' , 'f1'])
final_results = pd.DataFrame(np.transpose(means), columns=cv_df.model_name.unique())
final_results.index = ['Accuracy', 'Precision', 'Recall', 'F-measure'] 
print("")  
print(cv_df)
print("") 
print(final_results)

#Print boxplot for comparing models
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()



