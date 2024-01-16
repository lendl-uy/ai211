# Mathematical Blocks in the Transformer Architecture
# [1] InputEmbedding
# [2] PositionalEncoding
# [3] MultiHeadAttention
# [4] LayerNorm
# [5] FeedForward
# [6] LinearLayer

import numpy as np

def softmax(x, mask=None):
    if mask is not None:
        x = x * mask
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True)) # subtract np.max for numerical stability
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
class InputEmbedding:

    def __init__ (self, d_model, vocab_size):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.vocab_size = vocab_size

        glorot_scale = np.sqrt(2.0 / (vocab_size + d_model)) # common std for initialization
        self.vocab_embedding = np.random.normal(0,glorot_scale,(vocab_size,d_model))

        self.grad_vocab_embedding = np.zeros((vocab_size,d_model))

    def embedding(self, input_indices):
        input_indices = np.array(input_indices)
        embedded_output = self.vocab_embedding[input_indices]
        return embedded_output
    
    def __call__(self, input_indices):
        input_indices = np.array(input_indices)
        return self.embedding(input_indices) * np.sqrt(self.d_model)
    
    def backward(self, input_indices, grad_output):
        # Compute gradients for the vocab_embedding
        self.grad_vocab_embedding = np.zeros_like(self.vocab_embedding)
        np.add.at(self.grad_vocab_embedding, input_indices, grad_output)

        return self.grad_vocab_embedding
    
class PositionalEncoding:
        
    def __init__(self, d_model, seq_length):
        self.d_model = d_model
        self.seq_length = seq_length

        self.pos_embed = np.zeros((seq_length, d_model))
        pos = np.arange(seq_length).reshape((seq_length,1))
        denom = np.exp(np.arange(0,d_model,2, dtype=np.float64) * (-np.log(10000.0)/d_model)) .reshape((int(d_model/2),1))

        # print("\npos shape: \n",pos.shape)
        # print("\ndenom shape: \n",denom.shape)

        self.pos_embed[:, 0::2] = np.sin(pos @ denom.T)
        self.pos_embed[:, 1::2] = np.cos(pos @ denom.T)

        # print("pos_embed: \n",self.pos_embed)
        # print("pos embed shape: \n",self.pos_embed.shape)

    def __call__(self,input_embeddings):
        pos_embed_3d = np.tile(self.pos_embed[np.newaxis, :, :], (input_embeddings.shape[0], 1, 1))
        pos_embed_3d = np.tile(self.pos_embed[np.newaxis, :, :], (input_embeddings.shape[0], 1, 1))
        input_embeddings += pos_embed_3d
        
        return input_embeddings

class MultiHeadAttention:

    def __init__(self, d_model, num_heads, masked = False): # d_model should be divisible by num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.masked = masked

        # Xavier Initialization of Weight Matrices
        # Suited for Layers with Softmax Functions
        # Create weight matrices w/ dim d_model x d_model and scaled by 1/sqrt(d_model) (common practice)
        self.W_q = np.random.randn(d_model,d_model) / np.sqrt(d_model)
        self.W_k = np.random.randn(d_model,d_model) / np.sqrt(d_model)
        self.W_v = np.random.randn(d_model,d_model) / np.sqrt(d_model)
        self.W_o = np.random.randn(d_model,d_model) / np.sqrt(d_model)

        # Create vars to store grads of the weight matrices
        self.grad_W_q = np.zeros((d_model,d_model))
        self.grad_W_k = np.zeros((d_model,d_model))
        self.grad_W_v = np.zeros((d_model,d_model))
        self.grad_W_o = np.zeros((d_model,d_model))

    def attention(self, query, key, value, masked = False):
        d_k = query.shape[-1]

        attention_scores = (query @ np.transpose(key, (0, 1, 3, 2))) / np.sqrt(d_k)

        if masked:
            seq_length = attention_scores.shape[-1]
            batch_size = attention_scores.shape[0]
            num_heads = attention_scores.shape[1]

            mask = np.triu(np.ones((seq_length, seq_length)))
            mask_4d = mask[:, :, np.newaxis, np.newaxis]
            mask_4d = np.tile(mask_4d, (1, 1, num_heads, batch_size))
            mask_4d = np.transpose(mask_4d, (3, 2, 0, 1))

            attention_scores = attention_scores * mask_4d
            attention_scores = np.where(mask_4d == 0, -np.inf, attention_scores)
            attention_scores = attention_scores + 1e-12 # for numerical stability

        attention_scores = softmax(attention_scores)
        return attention_scores @ value

    def __call__(self, q, k, v):
        self.query = q @ self.W_q
        self.key = k @ self.W_k
        self.value = v @ self.W_v

        query_4d = np.transpose(self.query.reshape(self.query.shape[0], self.query.shape[1], self.num_heads, self.d_k), (0, 2, 1, 3))
        key_4d = np.transpose(self.key.reshape(self.key.shape[0], self.key.shape[1], self.num_heads, self.d_k), (0, 2, 1, 3))
        value_4d = np.transpose(self.value.reshape(self.value.shape[0], self.value.shape[1], self.num_heads, self.d_k), (0, 2, 1, 3))

        attention_out = self.attention(query_4d, key_4d, value_4d, self.masked)
        attention_out = np.transpose(attention_out,(0, 2, 1, 3))
        self.attention_out = attention_out.reshape(attention_out.shape[0],attention_out.shape[1], self.d_model)

        return self.attention_out @ self.W_o
    
    def backward(self, grad_output):
        # Backward pass

        # Backward pass through the output weight matrix
        self.grad_W_o = self.attention_out.T @ grad_output
        grad_attention_scores = grad_output @ self.W_o.T

        # Backward pass through the softmax function
        grad_softmax = self.attention_out * (1.0 - self.attention_out)
        grad_attention_scores *= grad_softmax

        # Backward pass through the attention mechanism
        grad_query = grad_attention_scores @ np.transpose(self.W_k, (0, 1, 3, 2))
        grad_key = grad_attention_scores @ np.transpose(self.W_q, (0, 1, 3, 2))
        grad_value = grad_attention_scores @ np.transpose(self.W_v, (0, 1, 3, 2))

        # Backward pass through the weight matrices W_q, W_k, and W_v
        self.grad_W_q = self.query.T @ grad_query
        self.grad_W_k = self.key.T @ grad_key
        self.grad_W_v = self.value.T @ grad_value

        # Compute gradients for the input queries, keys, and values
        grad_query_input = grad_query @ self.W_q.T
        grad_key_input = grad_key @ self.W_k.T
        grad_value_input = grad_value @ self.W_v.T

        return grad_query_input, grad_key_input, grad_value_input

class LayerNorm:

    def __init__(self, d_model, eps = 10**-8):
        self.eps = eps
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

        self.grad_gamma = np.zeros(d_model)
        self.grad_beta = np.zeros(d_model)

    def __call__(self, input):
        self.input = input
        mean = np.mean(input, axis=-1, keepdims=True)
        std = np.std(input, axis=-1, keepdims=True)

        self.normalized_input = self.gamma * (input - mean) / (std + self.eps) + self.beta
        return self.normalized_input
    
    def backward(self, grad_output):
        # Backward pass

        # Compute gradients for gamma and beta
        self.grad_gamma = np.sum(grad_output * self.normalized_input, axis=-1, keepdims=True)
        self.grad_beta = np.sum(grad_output, axis=-1, keepdims=True)

        # Compute gradient of the loss with respect to the normalized input
        grad_normalized_input = grad_output * self.gamma

        # Compute gradients for mean and std
        mean = np.mean(self.input, axis=-1, keepdims=True)
        std = np.std(self.input, axis=-1, keepdims=True)
        diff_input_mean = self.input - mean
        grad_std = -0.5 * np.sum(grad_normalized_input * diff_input_mean / (std + self.eps)**3, axis=-1, keepdims=True)
        grad_mean = -np.sum(grad_normalized_input / (std + self.eps), axis=-1, keepdims=True) - 2.0 * grad_std * np.mean(diff_input_mean, axis=-1, keepdims=True)

        # Compute gradient of the loss with respect to the input
        grad_input = grad_normalized_input / (std + self.eps) + grad_std * 2.0 * diff_input_mean / self.input.shape[-1] + grad_mean / self.input.shape[-1]

        return grad_input
    
class FeedForward:
    def __init__(self, d_model, d_ff):
        # Weight matrix and bias for the 1st and the 2nd linear layer
        # He Intialization of Weight Matrices
        # Suitable for layers with ReLU
        self.linear1 = np.random.randn(d_model, d_ff) * np.sqrt(2/(d_model+d_ff))
        self.bias1 = np.zeros((1, d_ff))
        self.linear2 = np.random.randn(d_ff, d_model)  * np.sqrt(2/(d_model+d_ff))
        self.bias2 = np.zeros((1, d_model))

        # Create vars to store the gradients
        self.grad_linear1 = np.zeros((d_model,d_ff))
        self.grad_bias1 = np.zeros((1, d_ff))
        self.grad_linear2 = np.zeros((d_ff, d_model))
        self.grad_bias2 = np.zeros((1, d_model))

    def __call__(self, input):
        # First linear transformation
        linear_output1 = input @ self.linear1 + self.bias1

        # ReLU activation
        self.relu_output = np.maximum(linear_output1, 0)

        # Second linear transformation
        linear_output2 = self.relu_output @ self.linear2 + self.bias2

        return linear_output2
    
    def backward(self, input, grad_output):
        # Backward pass

        # Backward pass through the second linear layer
        self.grad_linear2 = self.relu_output.T @ grad_output
        self.grad_bias2 = np.sum(grad_output, axis=0, keepdims=True)
        grad_relu = grad_output.dot(self.linear2.T)

        # Backward pass through the ReLU activation
        grad_relu_input = grad_relu * (self.relu_output > 0)

        # Backward pass through the first linear layer
        self.grad_linear1 = input.T @ grad_relu_input
        self.grad_bias1 = np.sum(grad_relu_input, axis=0, keepdims=True)
        grad_input = grad_relu_input.dot(self.linear1.T)

        return grad_input
    
class LinearLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) / np.sqrt(input_size)
        self.bias = np.zeros((1, output_size))

        # Gradients
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)

    def __call__(self, input):
        self.input = input
        return input @ self.weights + self.bias

    def backward(self, grad_output):
        # Compute gradients
        batch_size = len(self.input)

        print(self.weights.shape)

        self.grad_weights = np.sum(self.input.transpose(0,2,1) * grad_output, axis=0) / batch_size
        self.grad_bias = np.sum(grad_output, axis=0, keepdims=True) / batch_size

        print(f'grad weights: \n {self.grad_weights.shape}')
        print(f'grad bias: \n{self.grad_bias}')

        # Backpropagate the gradient
        grad_input = grad_output * self.weights.T

        return grad_input

    def update_parameters(self, learning_rate):
        self.weights -= learning_rate * self.grad_weights
        self.bias -= learning_rate * self.grad_bias