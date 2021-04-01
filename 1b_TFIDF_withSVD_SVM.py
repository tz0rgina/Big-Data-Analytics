from sklearn.decomposition import TruncatedSVD

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import make_scorer
import seaborn as sns

#calculate execution time
import time
start_time = time.time()

#import pandas
import pandas as pd
print("imported pandas...")

# how many rows to import
number_of_rows = 100

print("now read train.csv")
#read train data
df_train=pd.read_csv("./train.csv" )
df_train.head()

df_train.Label.unique()
df_train.loc[0]['Title']
df_train.loc[0]['Content']
df_train.loc[0]['Label']

print("now read test.csv")
#read test data
df_test=pd.read_csv("./test.csv")
df_test.head()

df_test.Label.unique()
df_test.loc[0]['Title']
df_test.loc[0]['Content']
df_test.loc[0]['Label']
# REFERENCE: https://machinelearningmastery.com/singular-value-decomposition-for-dimensionality-reduction-in-python/?fbclid=IwAR06E9YU0nXXFB85_IilkLWHng6PGB7BrzacAxjqpxk8VsU8MPMhTNJSWSs

X_train = df_train.iloc[:, 1] + " " + df_train.iloc[:, 2]
X_test = df_test.iloc[:, 1] + " " + df_test.iloc[:, 2]

print("vectorize with TF-IDF -> BoW approach")
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(lowercase=True
                        #,max_features=10000,
                        # stop_words=stop_words,
                        # tokenizer=tokenizer.tokenize,
                        #max_df=0.2,
                        #min_df=0.02
                        )

# TFIDF for train data
data = tfidf.fit_transform(X_train)
df = pd.DataFrame(data[0].T.todense(), index=tfidf.get_feature_names(), columns=["TF-IDF"])
df = df.sort_values('TF-IDF', ascending=False)
print (df)
# TFIDF for test data
test_data = tfidf.transform(X_test)
test_df = pd.DataFrame(test_data[0].T.todense(), index=tfidf.get_feature_names(), columns=["TF-IDF"])
test_df = test_df.sort_values('TF-IDF', ascending=False)
print(test_df.head())

# number of SVD components
num_components = [5,10, 15, 25, 50, 100, 200, 400, 500, 1000, 1200, 2000]


# get a list of models to evaluate
def get_models():
    models = dict()
    for i in num_components:
        steps = [('svd', TruncatedSVD(n_components=i)), ('classification', SGDClassifier())]
        models[str(i)] = Pipeline(steps=steps)
    return models


# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
    custom_scorer = {'accuracy':make_scorer(accuracy_score),
                     'precision': make_scorer(precision_score, average='macro'),
                     'recall': make_scorer(recall_score, average='macro'),
                     'f1': make_scorer(f1_score, average='macro'),
                    }
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=0) # 5-fold cross validation
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=1) # , n_jobs=-1, error_score='raise'
    return scores

def crossValidation(X, y):

    print(y)
    print(X.shape, y.shape)
    # get the models to evaluate
    models = get_models()
    """
    # evaluate the models and store results
    results, names, mean_scores_list = list(), list(), list()
    for name, model in models.items():
        scores = evaluate_model(model, X, y)
        results.append(scores)
        names.append(name)
        mean_scores_list.append(mean(scores))
        print('> component %s %.3f (%.3f)' % (name, mean(scores), std(scores)))
    """
    
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
    
        print("Cross validation for SVM model with SVD with " + str(model_name) + " dimensions.")
        print("----------------------------------------------------------------")
        
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
        
    return means , entries

# define dataset
X = data
y = df_train['Label']
X_test=test_data
y_test = df_test['Label']
print(X.shape, y.shape)
print(X_test.shape, y_test.shape)

means, entries = crossValidation(X,y)
print (means)
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy', 'precision', 'recall' , 'f1'])
results = pd.DataFrame(np.transpose(means), columns=cv_df.model_name.unique())
results.index = ['Accuracy', 'Precision', 'Recall', 'F-measure'] 

print(cv_df)
print(results)

sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()
"""
cv_df.groupby('model_name').accuracy.mean()
# plot model performance for comparison
plt.figure(figsize=(8, 8), facecolor=None)
plt.boxplot(results, labels=names, showmeans=True)
plt.xticks(rotation=45)
# plt.show()
plt.savefig('Figure1_SVD_SVM.png')

# make final predictions using svd with classification
# define the model
steps = [('svd', TruncatedSVD(n_components=num_components)), ('classification', SGDClassifier())]
model = Pipeline(steps=steps)
model.fit(X, y)
print(X.shape, y.shape)
print(test_data.shape)
yhat = model.predict(test_data)

print(classification_report(y_test, yhat))
"""
"""
# make final KAGGLE predictions using svd with classification
print("now make KAGGLE predictions")
# UNLABELED TEST DATA
df_test_final=pd.read_csv("./datasets2020/q1/test_without_labels.csv")
df_test_final.head()
df_test_final.loc[0]['Title']
df_test_final.loc[0]['Content']
X_test_final = df_test_final.iloc[:, 1].values + df_test_final.iloc[:, 2].values  # HERE concatenate Title with Content

# TFIDF for test data
tfidf_test_final_sparse = tfidf.transform(X_test_final)
tfidf_test_final_df = pd.DataFrame(tfidf_test_final_sparse.toarray(), columns=tfidf.get_feature_names())
tfidf_test_final_df.head()

test_data_final = tfidf_test_final_df

# define the model
steps = [('svd', TruncatedSVD(n_components=num_components)), ('classification', svm.SVC())]
model = Pipeline(steps=steps)
model.fit(X, y)
yhat_pred_kaggle = model.predict(test_data_final)

#EXPORT RESUTLS for KAGGLE
print("export results to .csv")
resultsSVD_SVM = pd.DataFrame({'Id':df_test_final['Id'], 'Predicted':yhat_pred_kaggle})
resultsSVD_SVM.to_csv('testSet_categories_BoW_SVM.csv', index=False, header=True)

print("--- %s seconds ---" % (time.time() - start_time))

# TODO macro statistics
# print tables directly for report
"""