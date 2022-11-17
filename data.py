import pickle
import numpy as np
import os
import tensorflow.keras.utils as np_utils
import tarfile
import wget

def load_cifar_data(folder,tiny=False, download=False):

    if download:
        wget.download('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
        data_zip = os.path.join(os.getcwd(),'cifar-10-python.tar.gz')
        f = tarfile.open(data_zip)
        f.extractall(os.getcwd()) 
        f.close()
        os.remove(data_zip)

    train_batches = [f'{folder}/{f}' for f in os.listdir(folder) if 'batch_' in f]
    test_batch    =  f'{folder}/test_batch'

    # Get train data
    X_trn = None
    y_trn = []
    for i in range(len(train_batches)):
        train_data_dict = pickle.load(open(train_batches[i],'rb'), encoding='latin-1')
        if i+1 == 1:
            X_trn = train_data_dict['data']
        else:
            X_trn = np.vstack((X_trn, train_data_dict['data']))
        y_trn += train_data_dict['labels']
    X_trn = X_trn.reshape(len(X_trn),3,32,32)
    X_trn = np.rollaxis(X_trn,1,4)
    X_trn = X_trn.astype('float32')/255.0
    y_trn = np_utils.to_categorical(np.asarray(y_trn),10)

    # Get test data
    test_data_dict  = pickle.load(open(test_batch,'rb'), encoding='latin-1')
    X_tst = test_data_dict['data']
    X_tst = X_tst.reshape(len(X_tst),3,32,32)
    X_tst = np.rollaxis(X_tst,1,4)
    X_tst = X_tst.astype('float32')/255.0
    y_tst = np_utils.to_categorical(np.asarray(test_data_dict['labels']))
    
    n_90 = int(0.9*len(X_trn))
    X_trn, X_val = X_trn[:n_90], X_trn[n_90:]
    y_trn, y_val = y_trn[:n_90], y_trn[n_90:]

    if tiny:
        X_trn,y_trn,X_tst,y_tst,X_val,y_val = X_trn[:1000],y_trn[:1000],X_tst[:100],y_tst[:100],X_val[:100],y_val[:100]

    return X_trn, y_trn, X_tst, y_tst, X_val, y_val
