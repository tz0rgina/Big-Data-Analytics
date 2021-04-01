from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Lambda,TimeDistributed ,Input, Dropout, Flatten, BatchNormalization
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import sequence,text
from keras.layers.core import Activation,Dense,SpatialDropout1D
from keras.optimizers import Adam
from keras.layers.merge import concatenate
import keras.layers as layers
import logging
import keras
import multiprocessing
import gensim
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras import backend as K
import tensorflow as tf
import sys
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import make_scorer
import os

#----Global Variables-----#

MAX_LEN = 40

MAX_NB_WORDS = 200000
MAX_SEQUENCE_LENGTH = 25
EMBEDDING_DIM = 300
MODEL_WEIGHTS_FILE = 'siamese_weights.h5'
VALIDATION_SPLIT = 0.1
NB_EPOCHS = 8 #15
DROPOUT = 0.01
SPATIAL_DROPOUT = 0.2
BATCH_SIZE = 384
SPLITS = 5
OPTIMIZER = 'adam'

gen_model = KeyedVectors.load_word2vec_format((datapath('wiki.en.vec')))

def saveImage(save_name, name_folder):
    save_path = os.path.join(os.path.join(os.getcwd(), name_folder))
    if os.path.isdir(save_path):
        plt.savefig(os.path.join(save_path, save_name))
    else:
        os.mkdir(save_path)
        plt.savefig(os.path.join(save_path, save_name))
    return
    
def Angle(inputs):

     length_input_1=K.sqrt(K.sum(tf.pow(inputs[0],2),axis=1,keepdims=True))
     length_input_2=K.sqrt(K.sum(tf.pow(inputs[1],2),axis=1,keepdims=True))
     result=K.batch_dot(inputs[0],inputs[1],axes=1)/(length_input_1*length_input_2)
     angle = tf.acos(result)
     return angle

def Distance(inputs):

    s = inputs[0] - inputs[1]
    print(s)
    output = K.sum(s ** 2,axis=1,keepdims=True)
    return output
    
def custom_layer_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],1)

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
    
def prepare_data_as_inputs(data , tk):
    
    question1 = tk.texts_to_sequences(data.Question1.values)
    question1 = sequence.pad_sequences(question1,maxlen=MAX_LEN)

    question2 = tk.texts_to_sequences(data.Question2.values)
    question2 = sequence.pad_sequences(question2,maxlen=MAX_LEN)
    
    return question1 , question2

def build_siamese(word_index):
    
    K.clear_session()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    num_workers = multiprocessing.cpu_count()
    context_size = 5
    downsampling = 7.5e-06
    seed = 1
    min_word_count = 5
    hs = 1
    negative = 5
    gradient_clipping_norm = 1.25

    Quora_word2vec = gensim.models.Word2Vec(

        sg=0,
        seed=1,
        workers=num_workers,
        min_count=min_word_count,
        size=EMBEDDING_DIM,
        window=context_size,  
        hs=hs, 
        negative=negative,  
        sample=downsampling 

    )

    Quora_word2vec = gen_model
    embedding_matrix=np.zeros((len(word_index)+1,EMBEDDING_DIM))

    for word , i in tqdm(word_index.items()): #i is index

        try:

            embedding_vector =  Quora_word2vec[word] #Exception is thrown if there is key error
            embedding_matrix[i] = embedding_vector

        except Exception as e:  #If word is not found continue

            continue
        
    #Building the model
      
    #--------question1--------#

    model1_in = Input(shape=(MAX_LEN,))
    model1= Embedding(
        len(word_index)+1,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_LEN,
        trainable=False
        )(model1_in)

    model1 = SpatialDropout1D(SPATIAL_DROPOUT)(model1)
    model1 = TimeDistributed(Dense(EMBEDDING_DIM, activation='relu'))(model1)
    model1 = Lambda(lambda x: K.sum(x, axis=1), output_shape=(EMBEDDING_DIM,))(model1)
    

   #---------question2-------#
    
    model2_in = Input(shape=(MAX_LEN,))
    model2= Embedding(
        len(word_index)+1,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_LEN,
        trainable=False
        )(model2_in)

    model2 = SpatialDropout1D(SPATIAL_DROPOUT)(model2)
    model2 = TimeDistributed(Dense(EMBEDDING_DIM, activation='relu'))(model2)
    model2 = Lambda(lambda x: K.sum(x, axis=1), output_shape=(EMBEDDING_DIM,))(model2)

    #--------Compute-------#
    #-------Distances------#
    
    #Calculate distance between vectors
    Distance_merged_model = Lambda(Distance, output_shape=custom_layer_output_shape)([model1, model2]) 

    #Calculate Cosine Similarity between vectors
    Angle_merged_model = Lambda(Angle, output_shape=custom_layer_output_shape)([model1, model2])

    #---------Merged------#
    merged = layers.concatenate([Distance_merged_model,Angle_merged_model])
    merged = Dense(100, activation='relu' , name='conc_layer')(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.01)(merged)
    out = Dense(1, activation="sigmoid", name = 'out')(merged)

    model = Model(inputs=[model1_in, model2_in], outputs=out)

    print (model.summary())

    return model
 
def fit_all_data_predict_ublabeled( Q1_train,Q2_train , train_labels ,Q1_unlabeld,Q2_unlabeld, word_index):

    model = build_siamese(word_index)
    
    start_time = time.time()
    
    model.compile(loss='binary_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
    
    history = model.fit([Q1_train,Q2_train],y=train_labels, batch_size=BATCH_SIZE, epochs=NB_EPOCHS,
                     verbose=1,  shuffle=True)  
    
    model.save('siamese_with_feature_extraction.h5') #saving the model
    
    print("Time for training with all training set ") 
    print("--- %s seconds ---" % (time.time() - start_time))
    print("") 
 
    # make final KAGGLE predictions with classification
    yhat_pred_kaggle = make_prediction(model, [Q1_unlabeld,Q2_unlabeld])

    return yhat_pred_kaggle
    
    
    
def fit_and_evaluate_model(model , Q1_train,Q2_train , train_labels , Q1_test,Q2_test ,test_labels):

    model.compile(loss='binary_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
    
    early_stop=keras.callbacks.EarlyStopping(patience=2, monitor='val_accuracy' , restore_best_weights=True)
    
    history = model.fit([Q1_train,Q2_train],y=train_labels, batch_size=BATCH_SIZE, epochs=NB_EPOCHS,
                     verbose=1, validation_split=VALIDATION_SPLIT, shuffle=True, callbacks=[early_stop])
    
    score = model.evaluate([Q1_test,Q2_test], test_labels, verbose=0)
    
    predicted_labels = make_prediction(model, [Q1_test,Q2_test])

    #print(predicted_labels)

    actual_labels=pd.Series(test_labels, name="Actual")	
    predicted_labels=pd.Series(predicted_labels, name="Predicted")	
    #print(actual_labels)
    #print(predicted_labels)
    accuracy=score[1]
    f1 = 100*f1_score(test_labels, predicted_labels , average = 'macro')
    recall = 100*recall_score(test_labels, predicted_labels, average = 'macro')
    precision = 100 *precision_score(test_labels , predicted_labels, average = 'macro')

    print("Model stopped at epoch : " + str(len(history.history['accuracy'])))
    
    return accuracy, recall , precision , f1 , history

def crossValidation(Q1 , Q2 , y , splits,  fig_title , word_index):

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
        
        model = build_siamese(word_index)
        
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
    
    #IMPORT DATASET AND PREPROCESS 
    train = pd.read_csv("C:/BIG DATA/q2b/train.csv")
    filtered_data=Data_Cleaning(train)
    y=filtered_data.IsDuplicate.values
    
    test = pd.read_csv("C:/BIG DATA/q2b/test_without_labels.csv")
    unlabeled_data = Data_Cleaning(test)
    
    tk=text.Tokenizer()
    tk.fit_on_texts(list(filtered_data.Question1.values)+list(filtered_data.Question2.values))
    word_index = tk.word_index
    question1 , question2  = prepare_data_as_inputs(filtered_data , tk)
    question1_unlabeled , question2_unlabeled   = prepare_data_as_inputs(unlabeled_data, tk)
    
    #CROSS VALIDATION
    fig_title = "Accuracy and loss during 5 folds"
    crossValidation(question1 , question2 , y , splits=SPLITS, fig_title, word_index)
    
    #PREDICT UBLABEKED DATA
    yhat_pred_kaggle = fit_all_data_predict_ublabeled(question1 , question2 , y, question1_unlabeled , question2_unlabeled , word_index)
    
    #EXPORT RESUTLS for KAGGLE
    print("export results to .csv")
    results = pd.DataFrame({'Id':test['Id'], 'Expected':yhat_pred_kaggle})
    results.to_csv('2b_siamese_with_features.csv', index=False, header=True)

Main()
"""
  accuracy     recall  precision         f1
0  0.802601  79.820084  78.909574  79.263537
1  0.806717  78.586050  79.458844  78.965359
2  0.794527  79.310893  78.161726  78.547088
3  0.797088  79.519798  78.350693  78.753192
4  0.789823  79.074993  77.656524  78.068426

Metrics Statistics
----------------------------------------------------------------------------------------
Mean : [accuracy , recall , precision , f1] = [0.7981512308120727, 79.2623636554417, 78.5074720378406, 78.71952039310695]
"""