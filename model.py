import numpy as np
import util
import layers

class Model:
    def __init__(self,loss_name): 
        self.layers = []
        self.loss_dict = {'cat_cross_entropy':    {'forward':  util.cat_cross_entropy,
                                      'backward': util.cat_cross_entropy_prime}}
        self.loss_fwd = self.loss_dict[loss_name]['forward']

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, input_data):
        y_hat = []
        Z = input_data
        for layer in self.layers:
            Z = layer.forward(Z,init_weights=False)
        y_hat = Z
        return y_hat

    def fit(self, x_train, y_train, epochs, batch_size, learning_rate, x_val=None, y_val=None):
        history = {'accuracy': [],'loss': [],'val_accuracy': [],'val_loss': []}

        for e in range(epochs):
            print(e)

            loss_,acc,val_loss,val_acc=0,0,0,0

            mini_batches = util.get_mini_batches(x_train,y_train, batch_size)


            for i, mini_batch in enumerate(mini_batches):    
                print(f"Processing Batch {i}/{len(mini_batches)}", end='\r')

                x_batch = mini_batch[0]
                y_batch = mini_batch[1]

                # forward
                Z = x_batch

                for layer in self.layers:
                    init_weights=True if e==0 else False
                    Z = layer.forward(Z,init_weights=init_weights)

                y_real = y_batch
                y = np.argmax(y_real,axis=1)
                y_pred = Z


                # backward - dCCE/dsoftmax
                error = -y/(np.argmax(y_pred,axis=1) + util.eps())

                for layer in (self.layers)[::-1]:
                    error = layer.backward(error, learning_rate)
              
            loss_ /= x_train.shape[0]
            acc  /= x_train.shape[0]
            
            history['loss'].append(loss_)
            history['accuracy'].append(acc)

            if x_val is None or y_val is None:
                print(f'Epoch: {e}   loss = {str(round(loss_,3))}   acc = {str(round(acc,3))}')

            else:
                Z_val = x_val
                for layer in self.layers:
                   Z_val = layer.forward(Z_val,init_weights=False)

                y_real_val = y_val
                y_pred_val = Z_val

                val_loss = self.loss_fwd(y_real_val, y_pred_val)
                val_acc = sum(np.where(np.argmax(y_real_val,axis=1)==np.argmax(y_pred_val,axis=1),1,0))
                val_acc  /= x_val.shape[0]                

                history['val_accuracy'].append(val_acc)
                history['val_loss'].append(val_loss)

                print(f'Epoch: {e}   loss = {str(round(loss_,3))}   acc = {str(round(acc,3))}   val_loss = {str(round(val_loss,3))}   val_accuracy = {str(round(val_acc,3))}')

        return history
