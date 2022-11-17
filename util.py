import numpy as np

def L1(a,b):
    return np.abs(a-b)

def hard_tanh(array):
    return np.clip(array,-1,1)

def eps():
    return np.random.uniform(1e-07,1e-06)

def cat_cross_entropy(y_true, y_pred):    
    out = -np.sum(np.multiply(y_true,np.log(y_pred+eps())))
    return out/y_true.shape[0]

def cat_cross_entropy_prime(y_true,y_pred):
    return np.sum([-y/(yhat+eps()) for (y,yhat) in zip(y_true,y_pred)])

def get_mini_batches(X,y,batch_size):
    mini_batches = []
    for i in range(0,len(X), batch_size):
        lower = i
        upper = np.min([len(X), i + batch_size])
        X_batch = X[lower:upper]
        y_batch = y[lower:upper]
        mini_batches.append((X_batch,y_batch))

    return mini_batches
