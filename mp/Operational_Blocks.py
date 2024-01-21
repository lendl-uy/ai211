import numpy as np

from Transformer_Constants import *

np.set_printoptions(precision=2)

class InputEmbedding:

    def __init__ (self, d_model, vocab_size):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.vocab_size = vocab_size

        glorot_scale = np.sqrt(2.0 / (vocab_size + d_model)) # common std for initialization
        self.vocab_embedding = np.random.normal(0, glorot_scale, (vocab_size, d_model))

        self.grad_vocab_embedding = np.zeros((vocab_size, d_model))

    def embedding(self, input_indices):
        input_indices = np.array(input_indices)
        embedded_output = self.vocab_embedding[input_indices]

        return embedded_output
    
    def __call__(self, input_indices):
        input_indices = np.array(input_indices)
        return self.embedding(input_indices) * np.sqrt(self.d_model)
    
    def backward(self, input_indices, grad_upstreamient):
        # Compute gradients for the vocab_embedding
        self.grad_vocab_embedding = np.zeros_like(self.vocab_embedding)
        np.add.at(self.grad_vocab_embedding, input_indices, grad_upstreamient)
        
        return self.grad_vocab_embedding

class PositionalEncoding:
        
    def __init__(self, d_model, seq_length):
        self.d_model = d_model
        self.seq_length = seq_length

        pos_embedding = np.zeros((seq_length, d_model))
        pos = np.arange(seq_length).reshape((seq_length,1))
        denom = np.exp(np.arange(0,d_model,2, dtype=np.float32) * (-np.log(10000.0)/d_model))
        denom = denom.reshape((int(d_model/2),1))

        pos_embedding[:, 0::2] = np.sin(pos @ denom.T)
        pos_embedding[:, 1::2] = np.cos(pos @ denom.T)

        self.pos_embedding = pos_embedding[np.newaxis,:,:] # (1, seq_length, d_model)

    def forward(self, input_embeddings):
        input_embeddings += self.pos_embedding[:, :input_embeddings.shape[1]]
        return input_embeddings
    
    def backward(self):
        pass

class MultiHeadAttention:

    def __init__(self, d_model = D_MODEL, num_heads = HEADS, dropout = DROPOUT_PERCENT):
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        scaling = np.sqrt(2 / d_model)

        # Initialization of weights for the query, key, and value matrices as well
        # as weights for the attention matrix
        # Normal Xavier Initialization
        self.W_q = np.random.randn(d_model, d_model) / np.sqrt(d_model) * scaling
        self.W_k = np.random.randn(d_model, d_model) / np.sqrt(d_model) * scaling
        self.W_v = np.random.randn(d_model, d_model) / np.sqrt(d_model) * scaling
        self.W_o = np.random.randn(d_model, d_model) / np.sqrt(d_model) * scaling

        self.softmax = Softmax()

        # Create vars to store grads of the weight matrices and the softmax function
        self.grad_W_q = None
        self.grad_W_k = None
        self.grad_W_v = None
        self.grad_W_o = None

        self.grad_softmax = np.zeros((d_model, d_model))

    def split_heads_forward(self, X):
        batch_size = X.shape[0]
        return X.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
    
    def split_heads_backward(self, x):
        batch_size = x.shape[0]
        return x.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.num_heads * self.d_k)
    
    def group_heads_forward(self, X):
        batch_size = X.shape[0]
        return X.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.num_heads * self.d_k)
    
    def group_heads_backward(self, x):
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

    def forward(self, Q, K, V, mask):

        self.Q = Q.copy()
        self.K = K.copy()
        self.V = V.copy()

        Q_weighted = self.Q @ self.W_q
        K_weighted = self.K @ self.W_k
        V_weighted = self.V @ self.W_v

        # print(f"Q_weighted = {Q_weighted.shape}")
        # print(f"K_weighted = {K_weighted.shape}")
        # print(f"V_weighted = {V_weighted.shape}")
        
        # Split weighted Q, K, and V to "heads"
        self.Q_split = self.split_heads_forward(Q_weighted)
        self.K_split = self.split_heads_forward(K_weighted)
        self.V_split = self.split_heads_forward(V_weighted)

        # print(f"Q_split = {Q_split.shape}")
        # print(f"K_split = {K_split.shape}")
        # print(f"V_split = {V_split.shape}")

        # Compute the Attention matrix
        E = self.Q_split @ self.K_split.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)

        self.mask = np.asarray(mask)
        if self.mask is not None:
            self.mask = self.mask[:, np.newaxis, ...]
            E = np.where(self.mask == 0, float('-inf'), E) #float("-1e20")

        self.attention_scores = self.softmax.forward(E)
        attention_heads = self.attention_scores @ self.V_split

        # Group the attention heads
        self.concatenated_attention = self.group_heads_forward(attention_heads)

        O = self.concatenated_attention @ self.W_o

        return O
    
    def backward(self, grad_upstream):
        self.grad_W_o = np.sum(self.concatenated_attention.transpose(0, 2, 1) @ grad_upstream, axis = 0)

        grad_upstream = np.dot(grad_upstream, self.W_o.T)
        # print(f"O backward = {grad_upstream.shape}")   
        grad_upstream = self.group_heads_backward(grad_upstream)
        # print(f"group_heads_backward = {grad_upstream.shape}") 
        V_error = self.attention_scores.transpose(0, 1, 3, 2) @ grad_upstream
        # print(f"V backward = {V_error.shape}") 
        grad_upstream = grad_upstream @ self.V_split.transpose(0, 1, 3, 2)
        grad_upstream = self.softmax.backward(grad_upstream)
        # print(f"softmax backward = {grad_upstream.shape}") 

        if self.mask is not None:
            grad_upstream = np.where(self.mask == 0, 0, grad_upstream)

        grad_upstream = grad_upstream * 1/np.sqrt(self.d_k)

        Q_error = np.matmul(grad_upstream, self.K_split)
        # print(f"Q backward = {Q_error.shape}") 
        K_error = np.matmul(self.Q_split.transpose(0, 1, 3, 2), grad_upstream) #alter
        K_error = K_error.transpose(0, 1, 3, 2)
        # print(f"K backward = {K_error.shape}") 

        grad_Q = self.split_heads_backward(Q_error)
        grad_K = self.split_heads_backward(K_error)
        grad_V = self.split_heads_backward(V_error)
        # print(f"Q split backward = {grad_Q.shape}") 
        # print(f"K split backward = {grad_K.shape}") 
        # print(f"V split backward = {grad_V.shape}") 

        self.grad_W_q = np.sum(self.Q.transpose(0, 2, 1) @ grad_Q, axis = 0)
        self.grad_W_k = np.sum(self.K.transpose(0, 2, 1) @ grad_K, axis = 0)
        self.grad_W_v = np.sum(self.V.transpose(0, 2, 1) @ grad_V, axis = 0)

        grad_Q = np.dot(grad_Q, self.grad_W_q.T)
        grad_K = np.dot(grad_K, self.grad_W_k.T)
        grad_V = np.dot(grad_V, self.grad_W_v.T)

        # print(f"grad_Q = {grad_Q.shape}") 
        # print(f"grad_K = {grad_K.shape}") 
        # print(f"grad_V = {grad_V.shape}") 

        return grad_Q, grad_K, grad_V

class Normalization:

    def __init__(self, d_model, eps = 1e-10):
        self.eps = eps
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

        # self.grad_gamma = np.zeros(d_model)
        # self.grad_beta = np.zeros(d_model)

    def forward(self, X):
        self.input = X
        self.mean = np.mean(X, axis=-1, keepdims=True)
        self.std_dev = np.std(X, axis=-1, keepdims=True)

        self.normalized_input = (X - self.mean) / (self.std_dev + self.eps)
        self.X_out = self.gamma * self.normalized_input + self.beta
        return self.normalized_input.copy()

    def backward(self, grad_upstream):
        m = self.input.shape[-1]
        
        grad_gamma = np.sum(grad_upstream * self.normalized_input, axis=(0,1), keepdims=True)
        grad_beta = np.sum(grad_upstream, axis=(0,1), keepdims=True)

        self.grad_gamma = np.squeeze(grad_gamma)
        self.grad_beta = np.squeeze(grad_beta)

        grad_normalized_input = grad_upstream * self.gamma
        grad_std_dev = np.sum(grad_normalized_input * (self.input - self.mean) * -0.5 * (self.std_dev + self.eps)**(-1.5), axis=-1, keepdims=True)
        grad_mean = np.sum(grad_normalized_input * -1 / (self.std_dev + self.eps), axis=-1, keepdims=True) + grad_std_dev * np.sum(-2 * (self.input - self.mean), axis=-1, keepdims=True) / m

        grad_upstream = grad_normalized_input / (self.std_dev + self.eps) + grad_std_dev * 2 * (self.input - self.mean) / m + grad_mean / m

        return grad_upstream

class Feedforward:

    def __init__(self, d_model, d_ff):
        # Initialization of layer values
        self.input = None
        self.layer_1 = None
        self.layer_2 = None
        
        # Weight matrix and bias for the 1st and the 2nd linear layer
        # He Intialization of Weight Matrices
        # Suitable for layers with ReLU
        self.weights_1 = np.random.randn(d_model, d_ff) * np.sqrt(2/(d_model+d_ff))
        self.biases_1 = np.zeros((1, d_ff))
        self.weights_2 = np.random.randn(d_ff, d_model)  * np.sqrt(2/(d_model+d_ff))
        self.biases_2 = np.zeros((1, d_model))

        # Create vars to store the gradients
        self.grad_weights_1 = None
        self.grad_biases_1 = None
        self.grad_weights_2 = None
        self.grad_biases_2 = None

    def relu(self, x):
        return np.maximum(0, x)
        
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def forward(self, X):
        self.input = X
        # First linear transformation
        X_1 = X @ self.weights_1 + self.biases_1
        
        # ReLU activation
        self.layer_1 = self.relu(X_1)
        
        # Second linear transformation
        self.layer_2 = self.layer_1 @ self.weights_2 + self.biases_2

        return self.layer_2

    def backward(self, grad_upstream):
        self.grad_weights_2 = np.sum(self.layer_1.transpose(0, 2, 1) @ grad_upstream, axis = 0)
        self.grad_biases_2 = np.sum(grad_upstream, axis = (0, 1))

        grad_upstream = np.dot(grad_upstream, self.weights_2.T)

        self.grad_weights_1 = np.sum(self.input.transpose(0, 2, 1) @ grad_upstream, axis = 0)
        self.grad_biases_1 = np.sum(grad_upstream, axis = (0, 1))
        
        grad_upstream = np.dot(grad_upstream, self.weights_1.T)

        return grad_upstream

class Linear:

    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) / np.sqrt(2/(input_size+output_size))
        self.bias = np.zeros((1, output_size))

    def forward(self, x):
        self.input_data = x
        return x @ self.weights + self.bias

    def backward(self, grad_upstream):
        self.grad_weights = np.sum(self.input_data.transpose(0, 2, 1) @ grad_upstream, axis = 0)
        self.grad_bias = np.sum(grad_upstream, axis = (0, 1))

        output_error = np.dot(grad_upstream, self.weights.T)

        return output_error
    
    def update_parameters(self, learning_rate):
        self.weights -= learning_rate * self.grad_weights
        self.bias -= learning_rate * self.grad_bias


class Softmax:

    def forward(self, x):
        self.x = x
        shiftx = x - np.max(x)
        exps = np.exp(shiftx)
        self.output = exps/ np.sum(exps)

        return self.output
    
    def backward(self, dout):
        if len(self.x.shape) == 3:
            batch_size, seq_length, vocab_size = self.x.shape
            num_heads = 1  # Set num_heads to 1 for 3D tensors
            dout_reshaped = dout.reshape(batch_size, seq_length, vocab_size)
        elif len(self.x.shape) == 4:
            batch_size, num_heads, seq_length, _ = self.x.shape
            dout_reshaped = dout.reshape(batch_size, num_heads, seq_length, seq_length)

        # Compute the gradient of the softmax
        dx = np.zeros_like(self.output)

        for i in range(batch_size):
            for h in range(num_heads):
                for t in range(seq_length):
                    if len(self.x.shape) == 3:
                        softmax_output = self.output[i, t, :]
                    elif len(self.x.shape) == 4:
                        softmax_output = self.output[i, h, t, :]
                    jacobian = np.diag(softmax_output) - np.outer(softmax_output, softmax_output)

                    if len(self.x.shape) == 3:
                        dx[i, t, :] = np.dot(dout_reshaped[i, t, :], jacobian)
                    elif len(self.x.shape) == 4:
                        dx[i, h, t, :] = np.dot(dout_reshaped[i, h, t, :], jacobian)

        return dx

    # def forward(self, x):
    #     self.x = x
    #     return 1 / (1 + np.exp(-x))

    # def backward(self, grad):
    #     x = self.x
    #     f_x = self.forward(self.x)

    #     return grad * (f_x * (1.0 - f_x))
    
class LogSoftmax:
    def __init__(self):
        self.axis = -1

    def softmax_forward(self, x):
        e_x = np.exp(x - np.max(x, axis = self.axis, keepdims=True))
        
        self.softmax =  e_x / np.sum(e_x, axis = self.axis, keepdims=True)
        return self.softmax

    def forward(self, x):
        self.x = x
        self.log_softmax = np.log(self.softmax_forward(x))
        return self.log_softmax
    
    def backward(self, grad = None):
        batch_size = self.x.shape[0]
        softmax = self.softmax_forward(self.x)

        input_grad = grad - softmax * grad.sum(axis = self.axis, keepdims=True)

        return input_grad / batch_size
    
class CrossEntropy():
    def __init__(self, ignore_index = None) -> None:
        self.ignore_index = ignore_index
        self.log_softmax = LogSoftmax()

    def loss(self, y_pred, y_true):
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true, dtype=np.int32)
        log_softmax = self.log_softmax.forward(y_pred)
        nll_loss = -log_softmax[np.arange(len(y_true)), y_true]
        
        return np.where(y_true == self.ignore_index, 0, nll_loss)

    def derivative(self, y_pred, y_true):
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        batch_size = y_pred.shape[0]
        err = 1/batch_size
        nll_loss_der = -1 * np.where(np.isin(y_pred, y_pred[np.arange(len(y_true)), y_true]), err, 0).astype(y_pred.dtype)
       
        output_err = self.log_softmax.backward(nll_loss_der)
        
        return np.where(y_true.reshape(-1, 1) == self.ignore_index, 0, output_err)