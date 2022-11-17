import numpy as np
import tensorflow as tf
from util import eps, hard_tanh, L1


class Layer:

    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, X, init_weights):
        raise NotImplementedError

    def backward(self, upstream_g, learning_rate):
        raise NotImplementedError


class adder_layer(Layer):

    def __init__(self,output_channels,kernel_size=3,stride=1,padding=0,adaptive_eta=0):
        self.output_channels = output_channels
        self.output_channels = output_channels
        self.adaptive_eta=adaptive_eta
        self.kernel_size=kernel_size        
        self.stride = stride
        self.padding = padding
        self.filters = np.ones((self.output_channels,self.kernel_size,self.kernel_size,1))
        self.bias = np.zeros((self.output_channels,1,1,1))
        self.name = 'adder'

    def get_adaptive_lr(self, k, dfilters, eta):
        """    
        k           -- n_tensors 
        dfilters    -- c_out x k_H x k_W x c_in
        eta         -- scalar
        """
        norm = np.linalg.norm(dfilters, ord=2, axis=0)
        return (eta * np.sqrt(k)) / (norm+eps())

    def forward(self,X, init_weights=False):
        """    
        X       -- n_tensors x H x W x c_in
        filters -- c_out x k_H x k_W x c_in
        b       -- c_out x 1 x 1 x 1
        Z       -- n_tensors x H_new x W_new, c_out
        cache   -- info needed for backward pass
        """

        self.input = X
        self.input_channels = X.shape[-1]

        if init_weights==True:
            self.filters = np.random.normal(loc=0,scale=1,size=(self.output_channels, self.kernel_size, self.kernel_size, self.input_channels))
            self.bias = np.zeros((self.output_channels,1,1,self.input_channels))

        filters, bias, stride,padding = self.filters, self.bias, self.stride, self.padding

        n_tensors, H,   W,   c_in = X.shape
        c_out,     H_k, W_k, c_in = filters.shape
        n_filters = c_out

        X_padded = np.pad(X, ((0,0), (padding,padding), (padding,padding), (0,0)), 'constant', constant_values = (0,0))
        H_new = int((H + 2*padding - H_k)/stride)+1
        W_new = int((W + 2*padding - W_k)/stride)+1

        Z = np.zeros([n_tensors, H_new, W_new, c_out])

        for i in range(n_tensors):           # traverse batch
            this_img = X_padded[i,:,:,:]     # select ith image in batch
            for f in range(n_filters):       # traverse filters
                this_filter = filters[f,:,:,:]
                this_bias = bias[f,:,:,:]
                for h in range(H_new-H_k):   # traverse height
                    for w in range(W_new):   # traverse width
                        v0,v1 = h*stride, h*stride + H_k
                        h0,h1 = w*stride, w*stride + W_k
                        this_window = this_img[v0:v1,h0:h1,:]

                        Z[i, h, w, f] = np.abs(this_window-this_filter).sum()

        assert Z.shape == (n_tensors, H_new, W_new, n_filters)

        self.output = Z
        self.cache = X, filters, bias, stride, padding
        
        return self.output

    def backward(self, upstream_g, learning_rate):
        """
        upstream_g (dL/dZ) -- n_tensors x H_up x W_up x c_up
        cache (values from previous layers) -- (X, W, B, s, p)               
        
        Output:
        dX -- dL/dX, shape n_tensors x H_down x W_down x c_down
        dF -- dL/dW, shape n_filters x k x k x k
        dB -- dL/dB, shape n_filters x 1 x 1 x 1
        """
        
        X, filters, bias, stride, padding = self.cache

        n_tensors, H_down, W_down, c_down = X.shape
        n_filters, H_k,    W_k,    c_down = filters.shape
        n_tensors, H_up,   W_up,   c_up   = upstream_g.shape
        
        dX       = np.zeros_like(X)                           
        dfilters = np.zeros_like(filters)
        dbias    = np.zeros_like(bias)

        X_padded  = np.pad(X,  ((0,0), (padding,padding), (padding,padding), (0,0)), 'constant', constant_values = (0,0))
        dX_padded = np.pad(dX, ((0,0), (padding,padding), (padding,padding), (0,0)), 'constant', constant_values = (0,0))

        for i in range(n_tensors):                       
            x = X_padded[i]
            dx = dX_padded[i]
            
        #x, dx = X_padded, dX_padded
            for h in range(H_up):                   # traverse height
                for w in range(W_up):               # traverse width
                    for c in range(c_up):           # traverse filters
                        
                        v0,v1 = h,h+H_k
                        h0,h1 = w,w+W_k
                        
                        x_window = x[v0:v1, h0:h1, :]
                        f_window = filters[c,:,:,:]

                        dx_local = hard_tanh(f_window-x_window)
                        df_local = x_window-f_window

                        g = upstream_g[i, h, w, c]

                        dx[v0:v1, v0:v1, :] += dx_local * g
                        dfilters[c,:,:,:]   += df_local * g
                        dbias[c,:,:,:]      += g

            dX[i, :, :, :] = dx[padding:-padding, padding:-padding, :]

        assert(dX.shape == (n_tensors, H_down, W_down, c_down))

        adaptive_lr = self.get_adaptive_lr(n_filters, dfilters, self.adaptive_eta)

        self.filters -= learning_rate*adaptive_lr*dfilters
        self.bias    -= learning_rate*dbias

        return dX


class FullyConnected(Layer):

    def __init__(self,output_channels):
        super(Layer, self).__init__()
        self.output_channels = output_channels
        self.weights=np.ones((1,self.output_channels))
        self.bias=np.zeros((1,self.output_channels))
        self.name='fully connected'

    def forward(self, X,init_weights):
        self.input = X
        self.input_channels = X.shape[-1]
        if init_weights==True:
            self.weights = np.random.normal(loc=0,scale=1,size=(self.input_channels,self.output_channels))
            self.bias    = np.random.normal(loc=0,scale=1,size=(self.output_channels))

        self.output = np.dot(self.input, self.weights)
        for i in range(self.output.shape[0]):
            self.output[i] += self.bias
        return self.output

    def backward(self, upstream_g, learning_rate):
        dX    = np.dot(upstream_g, self.weights.T)
        dW    = np.dot(self.input.T, upstream_g)
        dbias = np.mean(upstream_g)
        self.weights -= learning_rate*dW
        self.bias    -= learning_rate*dbias

        return dX


class Flatten(Layer):

    def forward(self, X,init_weights):
        self.original_shape = X.shape
        self.output = X.reshape(X.shape[0], np.product(X.shape[1:]))
        self.name = 'flatten'
        return self.output

    def backward(self, upstream_g, learning_rate):
        dX = upstream_g.reshape(self.original_shape)
        return dX


class batch_norm_layer(Layer):

    def __init__(self):
        self.gamma = 1
        self.beta = 0
        self.name = 'Batch Norm'

    def forward(self, X,init_weights):
        """    
        X       -- n_tensors x H x W x c_in
        gamma   -- n_tensors x 1 x 1 x 1
        beta    -- n_tensors x 1 x 1 x 1
        cache   -- info needed for backward pass
        """

        self.input = X

        if init_weights==True:
            self.gamma = np.ones((1,1,1))
            self.beta = np.zeros((1,1,1))

        mean = np.mean(X,axis=(0, 1, 2), keepdims=True)
        var = np.mean(((X-mean)**2), axis=(0, 1, 2), keepdims=True)
        std = np.sqrt(var)
        
        X_center = X - mean
        X_norm = X_center/(std+eps())
        
        self.output = X_norm*self.gamma + self.beta
        self.cache = X, X_center, X_norm

        return self.output 

    def backward(self, upstream_g, learning_rate):
        """
        upstream_g (dL/dZ) -- n_tensors x H_up x W_up x c_up
        cache (values from previous layers) -- (X, X_norm)               
        
        Output:
        dX -- dL/dX, shape n_tensors x H_down x W_down x c_down
        dF -- dL/dW, shape n_filters x k x k x k
        dB -- dL/dB, shape n_filters x 1 x 1 x 1
        """

        X, X_center, X_norm = self.cache

        dGamma = np.sum(upstream_g * X_norm, axis=0)
        dBeta  = np.sum(upstream_g, axis=0)

        m = len(X)
        mean = np.mean(X)
        std = np.std(X)
        
        dX = np.zeros_like(X)

        for i in range(m):
            for j in range(m):
                dX[i] += (upstream_g[i] - upstream_g[j]*(1 + (X[i]-X[j])*(X[j]-mean)/std))

        dX *= self.gamma/((m**2)*std)
        
        self.gamma = self.gamma - learning_rate*dGamma
        self.beta  = self.beta  - learning_rate*dBeta

        return dX


class MaxPool(Layer):

    def __init__(self,pool_size=2):
        self.pool_size=pool_size
        self.stride = pool_size
        self.name = 'Max Pool'

    def forward(self,X,init_weights):

        n_tensors, H, W, c_in = X.shape

        H_new = int(1 + (H - self.pool_size) / self.stride)
        W_new = int(1 + (W - self.pool_size) / self.stride)
        c_out = c_in

        Z = np.zeros((n_tensors, H_new, W_new, c_out))              
        
        for i in range(n_tensors):                     # loop over the training examples
            for h in range(H_new):                     # loop on the vertical axis of the output volume
                for w in range(W_new):                 # loop on the horizontal axis of the output volume
                    for c in range(c_out):             # loop over the channels of the output volume
                        
                        v0,v1 = h*self.stride, h*self.stride + self.pool_size
                        h0,h1 = w*self.stride, w*self.stride + self.pool_size
                        
                        window = X[i, v0:v1, h0:h1,c]
                        Z[i, h, w, c] = np.max(window)
                    

        self.output = Z
        self.cache = X, self.pool_size, self.stride
        
        return self.output

    def backward(self, upstream_g,learning_rate):
        X, pool_size, stride = self.cache

        n_tensors, H_down, W_down, c_down = X.shape 
        n_tensors, H_up,   W_up,   c_up   = upstream_g.shape


        dX = np.zeros(X.shape)
        
        for i in range(n_tensors):                       
            x = X[i]
            for h in range(H_up):       
                for w in range(W_up):    
                    for c in range(c_up):       
                        v0,v1 = h, h+pool_size
                        h0,h1 = w, w+pool_size

                        x_window = x[v0:v1, h0:h1, c]
                        
                        local_g = np.where(x_window==np.max(x_window),1,0)
                        # single images:
                        # g = upstream_g[h,w,c] 
                        # batches: 
                        g = upstream_g[i, h, w, c]

                        # single images:  
                        # dX[v0:v1, h0:h1, c] += local_g*g  
                        # batches: 
                        dX[i, v0:v1, h0:h1, c] += local_g * g

        assert(dX.shape == X.shape)

        return dX


def relu_fwd(X):
    return np.where(X>=0,X,0)


def relu_bwd(X):
    return np.where(X>=0,1,0)


def softmax_fwd(x):
    soft = tf.nn.softmax(x)
    return soft.numpy()


def softmax_bwd(X): 
    s = softmax_fwd(X)
    J = np.zeros_like(s)

    for i in range(len(s)):
        for j in range(len(s[i])):
            indicator_ij = 1 if i==j else 0
            J[i][j] = s[i][j]*(indicator_ij - s[i][j])
    return J


class Activation(Layer):

    def __init__(self,activation_name):
        self.name = activation_name
        super(Layer, self).__init__()

    def forward(self, X, init_weights=False):
        self.input = X
        if self.name == 'relu':
            self.output = np.where(X>=0,X,0)
            return self.output
        elif self.name == 'softmax':
            self.output = tf.nn.softmax(X).numpy()
            return self.output

    def backward(self, upstream_g, learning_rate,y_true=None):

        local_g = None     
        if self.name == 'relu':
            local_g = np.where(self.input>=0,1,0)
            dX = learning_rate*upstream_g
            return dX

        elif self.name == 'softmax':
            y_true = self.zeros_like(self.output) if y_true is None else y_true
            upstream_g[range(y_true.shape[0]), np.argmax(y_true,axis=1)] -= 1
            local_g = np.ones_like(upstream_g)
            dX = learning_rate*upstream_g
            return dX


class conv_layer(Layer):

    def __init__(self,output_channels,kernel_size=3,stride=1,padding=0):
        self.output_channels = output_channels


        self.output_channels = output_channels
        self.adaptive_eta=0

        self.kernel_size=kernel_size        
        self.stride = stride
        self.padding = padding
        self.name = 'Conv'



    def forward(self,X):
        """    
        X       -- n_tensors x H x W x c_in
        filters -- c_out x k_H x k_W x c_in
        b       -- c_out x 1 x 1 x 1
        Z       -- n_tensors x H_new x W_new, c_out
        cache   -- info needed for backward pass
        """
        self.input = X

        # in case input size not given
        self.input_channels = X.shape[-1]

        self.filters = np.random.normal(loc=0,scale=1,size=(self.output_channels, self.kernel_size, self.kernel_size, self.input_channels))
        self.bias    = np.random.normal(loc=0,scale=1,size=(self.output_channels, 1,1 ,self.input_channels))
        
        filters,stride,padding,bias = self.filters, self.stride, self.padding, self.bias
        n_tensors, H,   W,   c_in = X.shape
        c_out,     H_k, W_k, c_in = filters.shape
        n_filters = c_out

        X_padded = np.pad(X, ((0,0), (padding,padding), (padding,padding), (0,0)), 'constant', constant_values = (0,0))
        H_new = int((H + 2*padding - H_k)/stride)+1
        W_new = int((W + 2*padding - W_k)/stride)+1

        Z = np.zeros([n_tensors, H_new, W_new, c_out])

        for i in range(n_tensors):           # traverse batch
            this_img = X_padded[i,:,:,:]     # select ith image in batch
            for f in range(n_filters):       # traverse filters
                this_filter = filters[f,:,:,:]
                this_bias   = bias[f,:,:,:]
                for h in range(H_new):       # traverse height
                    for w in range(W_new):   # traverse width
                        v0,v1 = h*stride, h*stride + H_k
                        h0,h1 = w*stride, w*stride + W_k
                        this_window = this_img[v0:v1,h0:h1,:]

                        Z[i, h, w, f] = np.sum((np.multiply(this_window,this_filter) + this_bias.astype(float))).astype(float)

        assert Z.shape == (n_tensors, H_new, W_new, n_filters)

        self.output = Z
        self.cache = X, filters, bias, stride, padding
        
        return self.output

    def backward(self, upstream_g, learning_rate):
        """
        upstream_g (dL/dZ) -- n_tensors x H_up x W_up x c_up
        cache (values from previous layers) -- (X, W, B, s, p)               
        
        Output:
        dX -- dL/dX, shape n_tensors x H_down x W_down x c_down
        dF -- dL/dW, shape n_filters x k x k x k
        dB -- dL/dB, shape n_filters x 1 x 1 x 1
        """
        X, filters, bias, stride, padding = self.cache

        n_tensors, H_down, W_down, c_down = X.shape
        n_filters, H_k,    W_k,    c_down = filters.shape
        n_tensors, H_up,   W_up,   c_up   = upstream_g.shape
        
        dX       = np.zeros_like(X)                           
        dfilters = np.zeros_like(filters)
        dbias    = np.zeros_like(bias)

        X_padded  = np.pad(X,  ((0,0), (padding,padding), (padding,padding), (0,0)), 'constant', constant_values = (0,0))
        dX_padded = np.pad(dX, ((0,0), (padding,padding), (padding,padding), (0,0)), 'constant', constant_values = (0,0))
        
        for i in range(n_tensors):                       
            x = X_padded[i]
            dx = dX_padded[i]
            
            for h in range(H_up):                   # traverse height
                for w in range(W_up):               # traverse width
                    for c in range(c_up):           # traverse filters
                        
                        v0,v1 = h,h+H_k
                        h0,h1 = w,w+W_k
                        
                        x_window = x[v0:v1, h0:h1, :]
                        f_window = filters[c,:,:,:]

                        dx_local = hard_tanh(f_window-x_window)
                        df_local = x_window-f_window

                        g = upstream_g[i, h, w, c]

                        dx[v0:v1, v0:v1, :] += np.multiply(dx_local,g)
                        dfilters[c,:,:,:]   += np.multiply(df_local,g)
                        dbias[c,:,:,:]      += g
                        
            dX[i, :, :, :] = dx[padding:-padding, padding:-padding, :]
        
        assert(dX.shape == (n_tensors, H_down, W_down, c_down))

        self.filters -= learning_rate*dfilters
        self.bias    -= learning_rate*dbias

        return dX
