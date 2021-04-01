from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


#calculate execution time
import time
start_time = time.time()

#import pandas
import pandas as pd
print("imported pandas...")

print("now read train.csv")
#read train data
df_train=pd.read_csv("./train.csv")
print(df_train.head())

df_train.Label.unique()
#df_train.loc[0]['Title']
#df_train.loc[0]['Content']
#df_train.loc[0]['Label']

print("now read test.csv")
#read test data
df_test=pd.read_csv("./test.csv")
print(df_test.head())

X_train = df_train.iloc[:, 1] + " " + df_train.iloc[:, 2]
print(X_train[1])
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
tfidf_train_sparse = tfidf.fit_transform(X_train)
print(tfidf_train_sparse[1])
df = pd.DataFrame(tfidf_train_sparse[0].T.todense(), index=tfidf.get_feature_names(), columns=["TF-IDF"])
df = df.sort_values('TF-IDF', ascending=False)
print (df)

# TFIDF for test data
tfidf_test_sparse = tfidf.transform(X_test)
test_df = pd.DataFrame(tfidf_test_sparse[0].T.todense(), index=tfidf.get_feature_names(), columns=["TF-IDF"])
test_df = test_df.sort_values('TF-IDF', ascending=False)
print(test_df.head())

X = tfidf_train_sparse
y = df_train['Label'] 

# Number of trees in random forest
a = [10 , 20 , 50 , 100 , 150]
b = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
n_estimators = np.concatenate((a, b), axis=0)

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()


# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42)

# Fit the random search model
rf_random.fit(X, y)

rf_random.best_params_
print(rf_random.best_params_)

print("")
print("End of hyperparameter tuning!!!")
print("")

X_test =tfidf_test_sparse 
y_test = df_test['Label']
 
base_model = RandomForestClassifier(random_state = 42)
base_model.fit(X, y)

yhat = base_model.predict(X_test)

print("Classification Report for base model.")
print("-------------------------------------")
print(classification_report(y_test, yhat))

best_random = rf_random.best_estimator_

yhat_best= best_random.predict(X_test)

print("Classification Report for best model.")
print("-------------------------------------")
print(classification_report(y_test, yhat_best))

