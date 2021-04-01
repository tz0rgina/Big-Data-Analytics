from __future__ import print_function
import numpy as np
import csv, datetime, time, json
from zipfile import ZipFile
from os.path import expanduser, exists
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, TimeDistributed, Dense, Lambda, concatenate, Dropout, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint
from keras import backend as K
import keras
import pandas as pd
import matplotlib.pyplot as plt
import decimal
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import os
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import make_scorer

#calculate execution time
import time
start_time = time.time()

#----Global Variables-----#
KERAS_DATASETS_DIR = expanduser('C:/BIG DATA/')
GLOVE_ZIP_FILE_URL = 'C:/BIG DATA/glove.840B.300d.zip'
GLOVE_ZIP_FILE = 'glove.840B.300d.zip'
GLOVE_FILE = 'glove.840B.300d.txt'
MAX_NB_WORDS = 200000
MAX_SEQUENCE_LENGTH = 25
EMBEDDING_DIM = 300
MODEL_WEIGHTS_FILE = 'siamese_deep_net_weights.h5'
VALIDATION_SPLIT = 0.1
SPLITS = 5
RNG_SEED = 13371447
NB_EPOCHS = 10
DROPOUT = 0.1
BATCH_SIZE = 32
OPTIMIZER = 'adam'

def saveImage(save_name, name_folder):
    save_path = os.path.join(os.path.join(os.getcwd(), name_folder))
    if os.path.isdir(save_path):
        plt.savefig(os.path.join(save_path, save_name))
    else:
        os.mkdir(save_path)
        plt.savefig(os.path.join(save_path, save_name))
    return

def Data_Cleaning(data):

    corpus_raw = data
    
    #Checking whether there are any rows with null values
    nan_rows = corpus_raw[corpus_raw.isnull().any(1)]
    print (nan_rows)
    
    # Filling the null values with ' '
    corpus_raw = corpus_raw.fillna('')
    nan_rows = corpus_raw[corpus_raw.isnull().any(1)]
    print (nan_rows)

    corpus_raw['Question1'] = corpus_raw['Question1'].apply(lambda row: str(row).lower())
    corpus_raw['Question2'] = corpus_raw['Question2'].apply(lambda row: str(row).lower())



    # preserve contractions and ingore punctuations occuring in sentences--- remove punctuation between words except "'"
    corpus_raw['Question1'] = corpus_raw['Question1'].apply(lambda row: "".join([s if s.isalpha()  or s.isdigit() or s=="'" or s==" " else ' ' for s in row ]))
    corpus_raw['Question2'] = corpus_raw['Question2'].apply(lambda row: "".join([s  if s.isalpha() or s.isdigit() or s == "'" or s == " " else ' ' for s in row]))


    # Remove stopwords , numeric and alphanumeric characters

    corpus_raw['Question1_tokens'] = corpus_raw['Question1'].apply(
        lambda row: [row for row in row.split() if row is not row.isalpha()])
    corpus_raw['Question2_tokens'] = corpus_raw['Question2'].apply(

        lambda row: [row for row in row.split() if row is not  row.isalpha()])

    corpus_raw['Question1_tokens'] = corpus_raw['Question1_tokens'].apply(lambda row: ",".join(row))
    corpus_raw['Question2_tokens'] = corpus_raw['Question2_tokens'].apply(lambda row: ",".join(row))


    return corpus_raw
    
train = pd.read_csv("C:/BIG DATA/q2b/train.csv")
filtered_data = Data_Cleaning(train)   


question1 = []
question2 = []
is_duplicate = []
count=1

for i in range(len(filtered_data['Id'])):
    # if count > 500: # read first 100 rows
    #     break
    question1.append(filtered_data['Question1'][i])
    question2.append(filtered_data['Question2'][i])
    is_duplicate.append(filtered_data['IsDuplicate'][i])
    count += 1

print('Question pairs: %d' % len(question1))

# Build tokenized word index
questions = question1 + question2
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(questions)
question1_word_sequences = tokenizer.texts_to_sequences(question1)
question2_word_sequences = tokenizer.texts_to_sequences(question2)
word_index = tokenizer.word_index

print("Words in index: %d" % len(word_index))

print("Processing", GLOVE_FILE)

embeddings_index = {}
with open(KERAS_DATASETS_DIR + GLOVE_FILE, encoding='utf-8') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embedding

print('Word embeddings: %d' % len(embeddings_index))

# Prepare word embedding matrix
nb_words = min(MAX_NB_WORDS, len(word_index))
word_embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        word_embedding_matrix[i] = embedding_vector

print('Null word embeddings: %d' % np.sum(np.sum(word_embedding_matrix, axis=1) == 0))

# Prepare training data tensors
question1 = pad_sequences(question1_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
question2 = pad_sequences(question2_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
y = np.array(is_duplicate, dtype=int)
print('Shape of question1 data tensor:', question1.shape)
print('Shape of question2 data tensor:', question2.shape)
print('Shape of label tensor:', y.shape)

test = pd.read_csv("C:/BIG DATA/q2b/test_without_labels.csv")
unlabeled_data = Data_Cleaning(test)

q1_unlabeled = []
q2_unlabeled = []

for i in range(len(unlabeled_data['Id'])):
    # if count > 500: # read first 100 rows
    #     break
    q1_unlabeled.append(unlabeled_data['Question1'][i])
    q2_unlabeled.append(unlabeled_data['Question2'][i])

print('Unlabeled Question pairs: %d' % len(q1_unlabeled))

# Build tokenized word index unlabeled
questions_unlabeled = q1_unlabeled + q2_unlabeled
question1_word_sequences_unlabeled = tokenizer.texts_to_sequences(q1_unlabeled)
question2_word_sequences_unlabeled = tokenizer.texts_to_sequences(q2_unlabeled)
word_index_unlabeled = tokenizer.word_index

print("Unlabeled Words in index: %d" % len(word_index_unlabeled))

# Prepare training data tensors
question1_unlabeled = pad_sequences(question1_word_sequences_unlabeled, maxlen=MAX_SEQUENCE_LENGTH)
question2_unlabeled = pad_sequences(question2_word_sequences_unlabeled, maxlen=MAX_SEQUENCE_LENGTH)

print('unlabeled Shape of question1 data tensor:', question1_unlabeled.shape)
print('unlabeled Shape of question2 data tensor:', question2_unlabeled.shape)
    
def make_prediction(model, X_test):
    pred = model.predict(X_test)
    """
    for i in range(len(X_test[0])):
        print("%f" % pred[i])
    """
    predicted_results = []
    for i in range(len(X_test[0])):
        if pred[i] < 0.5:
            predicted_results.append(0)
        elif pred[i] >= 0.5:
            predicted_results.append(1)
    return predicted_results
    

def build_siamese_deep_net():
    
    K.clear_session()

   # Define the model
    question1 = Input(shape=(MAX_SEQUENCE_LENGTH,))
    question2 = Input(shape=(MAX_SEQUENCE_LENGTH,))

    q1 = Embedding(nb_words + 1,
                   EMBEDDING_DIM,
                   weights=[word_embedding_matrix],
                   input_length=MAX_SEQUENCE_LENGTH,
                   trainable=False)(question1)
    q1 = TimeDistributed(Dense(EMBEDDING_DIM, activation='relu'))(q1)
    q1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM,))(q1)

    q2 = Embedding(nb_words + 1,
                   EMBEDDING_DIM,
                   weights=[word_embedding_matrix],
                   input_length=MAX_SEQUENCE_LENGTH,
                   trainable=False)(question2)
    q2 = TimeDistributed(Dense(EMBEDDING_DIM, activation='relu'))(q2)
    q2 = Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM,))(q2)

    merged = concatenate([q1, q2])
    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)

    is_duplicate = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[question1, question2], outputs=is_duplicate)
    
    model.summary()
    
    return model
 
def fit_all_data_predict_ublabeled( Q1_train,Q2_train , train_labels ,Q1_unlabeld,Q2_unlabeld):

    start_time = time.time()
    
    model = build_siamese_deep_net()
    
    model.compile(loss='binary_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

    history = model.fit([Q1_train, Q2_train],
                        train_labels,
                        epochs=NB_EPOCHS,
                        #validation_split=VALIDATION_SPLIT,
                        verbose=1,
                        batch_size=BATCH_SIZE)
                     
    model.save('siamese_deep_net)full_data.h5') #saving the model
                     
    
    print("Time for training with all training set ") 
    print("--- %s seconds ---" % (time.time() - start_time))
    print("")
 
    # make final KAGGLE predictions with classification
    yhat_pred_kaggle = make_prediction(model, [Q1_unlabeld,Q2_unlabeld])

    #EXPORT RESUTLS for KAGGLE
    print("export results to .csv")
    results = pd.DataFrame({'Id':test['Id'], 'Expected':yhat_pred_kaggle})
    results.to_csv('2b_siamese_deep_net.csv', index=False, header=True)
    
    
    
def fit_and_evaluate_model(model , Q1_train,Q2_train , train_labels , Q1_test,Q2_test ,test_labels):

    model.compile(loss='binary_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

    # Train the model, checkpointing weights with best validation accuracy
    early_stop=keras.callbacks.EarlyStopping(patience=2, monitor='val_accuracy' , restore_best_weights=True)
    #callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_accuracy', save_best_only=True)]
    history = model.fit([Q1_train, Q2_train],
                        train_labels,
                        epochs=NB_EPOCHS,
                        validation_split=VALIDATION_SPLIT,
                        verbose=1,
                        batch_size=BATCH_SIZE,
                        callbacks=early_stop)
    
    score = model.evaluate([Q1_test,Q2_test], test_labels, verbose=0)
    
    predicted_labels = make_prediction(model, [Q1_test,Q2_test])

    #print(predicted_labels)

    actual_labels=pd.Series(test_labels, name="Actual")	
    predicted_labels=pd.Series(predicted_labels, name="Predicted")	
    accuracy=score[1]
    f1 = 100*f1_score(test_labels, predicted_labels , average = 'macro')
    recall = 100*recall_score(test_labels, predicted_labels, average = 'macro')
    precision = 100 *precision_score(test_labels , predicted_labels, average = 'macro')

    print("Model stopped at epoch : " + str(len(history.history['accuracy'])))
    
    return accuracy, recall , precision , f1 , history

def crossValidation(Q1 , Q2 , y , splits,  fig_title):

    acc_list=[]
    precision_list=[]
    recall_list=[]
    f1_list=[]
    
    X = np.stack((Q1, Q2), axis=1)
    
    kf = KFold(n_splits=splits, random_state=0, shuffle=False)
    
    noOfFold = 1
    noOfPlot = 1
    
    fig = plt.figure(figsize=(6,14))
    plt.suptitle(fig_title , fontsize=14, fontweight='bold')
    
    for train_index, test_index in kf.split(y):
    
        #print("TRAIN:", train_index, "TEST:", test_index)
        train_questions, test_questions = np.array(X)[train_index], np.array(X)[test_index]
        train_labels , test_labels = np.array(y)[train_index], np.array(y)[test_index]
  
        Q1_train = train_questions[:, 0]
        Q2_train = train_questions[:, 1]
        Q1_test = test_questions[:, 0]
        Q2_test = test_questions[:, 1]
        
        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {noOfFold} ...')
        
        model = build_siamese_deep_net()
        
        accuracy, recall , precision , f1 , history= fit_and_evaluate_model(model , Q1_train, Q2_train, train_labels ,  Q1_test, Q2_test, test_labels )
        """
        plot_graphs(history, "accuracy")
        plot_graphs(history, "loss")
        """
        plot_title = "fold No : " + str(noOfFold) + " - f1 = " + str("%.3f" % f1)
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        x_axis = range(1, len(acc) + 1)
        
        plt.subplot(splits,2,noOfPlot)
        plt.tight_layout()
        plt.plot(x_axis , acc, 'bo', label='Training accuracy')
        plt.plot(x_axis , val_acc, 'b', label='Validation accuracy')
        plt.title(plot_title, fontsize='x-small')
        plt.xticks(fontsize='x-small')
        plt.yticks(fontsize='x-small')
        
        noOfPlot+=1
        plt.subplot(splits,2,noOfPlot)
        plt.tight_layout()
        plt.plot(x_axis , loss, 'bo', label='Training loss')
        plt.plot(x_axis , val_loss, 'b', label='Validation loss')
        plt.title(plot_title, fontsize='x-small')
        plt.xticks(fontsize='x-small')
        plt.yticks(fontsize='x-small')
        
        acc_list.append(accuracy)        
        recall_list.append(recall)
        precision_list.append(precision)
        f1_list.append(f1)
        print("[accuracy , recall , precision , f1] = " + str([accuracy , recall , precision , f1]))
        
        #plot_accuracy_and_loss(history, "fold No : " + str(noOfFold))
        #print("Model stopped at epoch : " + str(len(history.history['acc'])))
        noOfFold+=1
        noOfPlot+=1
        
    results = pd.DataFrame(data = {'accuracy' : acc_list, 'recall' : recall_list, 'precision' : precision_list, 'f1' : f1_list})
    print (results)
   
    sum_acc=np.mean(acc_list)    
    sum_recall = np.mean(recall_list)
    sum_precision = np.mean(precision_list)
    sum_f1 = np.mean(f1_list)
    
    print("")
    print("Metrics Statistics")
    print("----------------------------------------------------------------------------------------")
    print("Mean : [accuracy , recall , precision , f1] = " + str([sum_acc , sum_recall , sum_precision , sum_f1]))
    

    folder= 'C:/BIG DATA' 
    fname =  fig_title + '.png'
    print(fname)
    saveImage(fname , folder)
    plt.show()

    return acc_list , recall_list , precision_list , f1_list


def Main():

    splits=SPLITS
    fig_title = "Accuracy and loss during 5 folds - siamese deepnet"
    #crossValidation(question1 , question2 , y , splits , fig_title)
    fit_all_data_predict_ublabeled(question1 , question2 , y, question1_unlabeled , question2_unlabeled)

Main()

"""

   accuracy     recall  precision         f1
0  0.793661  78.028021  77.949356  77.988057
1  0.789633  77.112220  77.477681  77.282888
2  0.795286  78.199436  78.134210  78.166391
3  0.769580  75.481895  75.323674  75.399835
4  0.796502  78.304225  78.126525  78.212351

Metrics Statistics
----------------------------------------------------------------------------------------
Mean : [accuracy , recall , precision , f1] = [0.7889323353767395, 77.4251592557331, 77.40228893910053, 77.40990425285825]
"""