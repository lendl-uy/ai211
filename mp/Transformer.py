# AI 211 MACHINE PROBLEM 
# Code Authors
# [1] Jan Lendl R. Uy, 2019-00312
# [2] Ryan Roi Cayas, 2023-XXXXX
# Section: AI 211 FZZQ

# Useful references:

import numpy as np
np.set_printoptions(precision=4)

# Calling the class instance is equivalent to doing a forward pass

class Vocabulary:
    def __init__(self):
        self.token_to_index = {}
        self.index_to_token = {}
        self.next_index = 1

    def build_vocab(self, token):
        if token not in self.token_to_index:
            index = self.next_index
            self.token_to_index[token] = index
            self.index_to_token[index] = token
            self.next_index += 1

    # def get_index(self, token):
    #     return self.token_to_index.get(token, None)

    # def get_token(self, index):
    #     return self.index_to_token.get(index, None)

    def seq_to_idx(self, seqs, max_seq_length):
        seqs_indices = []
        for seq in seqs:
            indices = [self.token_to_index[token] for token in seq.split() if token in self.token_to_index]

            if len(indices) > max_seq_length:
                print(f"Sequence is too long. Max sequence length is {max_seq_length}.")

            padded_indices = indices + [0] * (max_seq_length - len(indices))
            seqs_indices.append(padded_indices)
        return seqs_indices
    
    def size(self):
        return len(self.token_to_index) + 1 # we add 1 for the padding

    def __call__(self,train_set):
        # build vocab from train set
        for sequence in train_set[:,0]:
            tokens = sequence.split()
            for token in tokens:
                self.build_vocab(token)




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





class MultiAttention:

    def __init__(self, d_model, num_heads, masked = False): # d_model should be divisible by num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.masked = masked

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
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True)) # subtract np.max for numerical stability
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


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

        attention_scores = self.softmax(attention_scores)
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
        self.grad_W_o = np.dot(self.attention_out.T, grad_output)
        grad_attention_scores = np.dot(grad_output, self.W_o.T)

        # Backward pass through the softmax function
        grad_softmax = self.attention_out * (1.0 - self.attention_out)
        grad_attention_scores *= grad_softmax

        # Backward pass through the attention mechanism
        grad_query = grad_attention_scores @ np.transpose(self.W_k, (0, 1, 3, 2))
        grad_key = grad_attention_scores @ np.transpose(self.W_q, (0, 1, 3, 2))
        grad_value = grad_attention_scores @ np.transpose(self.W_v, (0, 1, 3, 2))

        # Backward pass through the weight matrices W_q, W_k, and W_v
        self.grad_W_q = np.dot(self.query.T, grad_query)
        self.grad_W_k = np.dot(self.key.T, grad_key)
        self.grad_W_v = np.dot(self.value.T, grad_value)

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
        self.linear1 = np.random.randn(d_model, d_ff)  # Weight matrix for the first linear layer
        self.bias1 = np.zeros((1, d_ff))               # Bias for the first linear layer
        self.linear2 = np.random.randn(d_ff, d_model)  # Weight matrix for the second linear layer
        self.bias2 = np.zeros((1, d_model))            # Bias for the second linear layer

        # Create vars to store the gradients
        self.grad_linear1 = np.zeros((d_model,d_ff))
        self.grad_bias1 = np.zeros((1, d_ff))
        self.grad_linear2 = np.zeros((d_ff, d_model))
        self.grad_bias2 = np.zeros((1, d_model))

    def __call__(self, input):
        # First linear transformation
        linear_output1 = np.dot(input, self.linear1) + self.bias1

        # ReLU activation
        self.relu_output = np.maximum(linear_output1, 0)

        # Second linear transformation
        linear_output2 = np.dot(self.relu_output, self.linear2) + self.bias2

        return linear_output2
    
    def backward(self, input, grad_output):
        # Backward pass

        # Backward pass through the second linear layer
        self.grad_linear2 = np.dot(self.relu_output.T, grad_output)
        self.grad_bias2 = np.sum(grad_output, axis=0, keepdims=True)
        grad_relu = grad_output.dot(self.linear2.T)

        # Backward pass through the ReLU activation
        grad_relu_input = grad_relu * (self.relu_output > 0)

        # Backward pass through the first linear layer
        self.grad_linear1 = np.dot(input.T, grad_relu_input)
        self.grad_bias1 = np.sum(grad_relu_input, axis=0, keepdims=True)
        grad_input = grad_relu_input.dot(self.linear1.T)

        return grad_input
    
class EncoderBlock:
    def __init__(self, d_model, num_heads, d_ff):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.d_k = d_model // num_heads

        self.multi_attention = MultiAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm2 = LayerNorm(d_model)

    def backward(self, grad_decoder_block):
        # Backward pass through the decoder block

        # Backward pass through the layer normalization after the final linear layer
        grad_norm2 = self.norm2.backward(grad_decoder_block)

        # Backward pass through the feedforward layer
        grad_ff = self.feed_forward.backward(grad_norm2)

        # Reshape grad_ff to match the shape before layer normalization
        grad_ff_reshaped = grad_ff.reshape(grad_ff.shape[0], self.num_heads, self.d_k)

        # Backward pass through the layer normalization before the multi-head attention
        grad_norm1 = self.norm1.backward(grad_ff_reshaped)

        # Backward pass through the multi-head attention
        grad_multi_attention = self.multi_attention.backward(grad_norm1)

        return grad_multi_attention

    def update_parameters(self, learning_rate):
        # SGD update for each parameter in the encoder block

        # Multi-Head Attention
        self.multi_attention.W_q -= learning_rate * self.multi_attention.grad_W_q
        self.multi_attention.W_k -= learning_rate * self.multi_attention.grad_W_k
        self.multi_attention.W_v -= learning_rate * self.multi_attention.grad_W_v
        self.multi_attention.W_o -= learning_rate * self.multi_attention.grad_W_o

        # Layer Norm 1
        self.norm1.gamma -= learning_rate * self.norm1.grad_gamma
        self.norm1.beta -= learning_rate * self.norm1.grad_beta

        # Feed Forward
        self.feed_forward.linear1 -= learning_rate * self.feed_forward.grad_linear1
        self.feed_forward.linear2 -= learning_rate * self.feed_forward.grad_linear2
        self.feed_forward.bias1 -= learning_rate * self.feed_forward.grad_bias1
        self.feed_forward.bias2 -= learning_rate * self.feed_forward.grad_bias2

        # Layer Norm 2
        self.norm2.gamma -= learning_rate * self.norm2.grad_gamma
        self.norm2.beta -= learning_rate * self.norm2.grad_beta


    def __call__(self, input):
        # Multi-Head Self Attention
        attention_output = self.multi_attention(input, input, input)

        # Residual Connection and Normalization
        norm1_output = self.norm1(input + attention_output)

        # Feed Forward
        ff_output = self.feed_forward(norm1_output)

        # Residual Connection and Normalization
        encoder_output = self.norm2(norm1_output + ff_output)

        return encoder_output

class DecoderBlock:
    def __init__(self, d_model, num_heads, d_ff):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.d_k = d_model // num_heads

        self.masked_multi_attention = MultiAttention(d_model, num_heads, masked=True)
        self.norm1 = LayerNorm(d_model)
        self.multi_attention = MultiAttention(d_model, num_heads)
        self.norm2 = LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm3 = LayerNorm(d_model)

    def backward(self, grad_final_linear_layer):
        # Backward pass through the decoder block

        # Backward pass through the layer normalization after the final linear layer
        grad_norm3 = self.norm3.backward(grad_final_linear_layer)

        # Backward pass through the feedforward layer
        grad_ff = self.feed_forward.backward(grad_norm3)

        # Reshape grad_ff to match the shape before layer normalization
        grad_ff_reshaped = grad_ff.reshape(grad_ff.shape[0], self.num_heads, self.d_k)

        # Backward pass through the layer normalization before the multi-head attention
        grad_norm2 = self.norm2.backward(grad_ff_reshaped)

        # Backward pass through the multi-head attention
        grad_multi_attention = self.multi_attention.backward(grad_norm2)

        # Backward pass through the layer normalization before the masked multi-head attention
        grad_norm1 = self.norm1.backward(grad_multi_attention[0])

        # Backward pass through the masked multi-head attention
        grad_masked_multi_attention = self.masked_multi_attention.backward(grad_norm1)

        return grad_multi_attention[1], grad_masked_multi_attention[0] # first gradient is for encoder block, second one is for target embedding

    def update_parameters(self, learning_rate):
        # SGD update for each parameter in the encoder block

        # Masked Multi-Head Attention
        self.masked_multi_attention.W_q -= learning_rate * self.masked_multi_attention.grad_W_q
        self.masked_multi_attention.W_k -= learning_rate * self.masked_multi_attention.grad_W_k
        self.masked_multi_attention.W_v -= learning_rate * self.masked_multi_attention.grad_W_v
        self.masked_multi_attention.W_o -= learning_rate * self.masked_multi_attention.grad_W_o

        # Layer Norm 1
        self.norm1.gamma -= learning_rate * self.norm1.grad_gamma
        self.norm1.beta -= learning_rate * self.norm1.grad_beta

        # Multi-Head Attention
        self.multi_attention.W_q -= learning_rate * self.multi_attention.grad_W_q
        self.multi_attention.W_k -= learning_rate * self.multi_attention.grad_W_k
        self.multi_attention.W_v -= learning_rate * self.multi_attention.grad_W_v
        self.multi_attention.W_o -= learning_rate * self.multi_attention.grad_W_o

        # Layer Norm 2
        self.norm2.gamma -= learning_rate * self.norm2.grad_gamma
        self.norm2.beta -= learning_rate * self.norm2.grad_beta

        # Feed Forward
        self.feed_forward.linear1 -= learning_rate * self.feed_forward.grad_linear1
        self.feed_forward.linear2 -= learning_rate * self.feed_forward.grad_linear2
        self.feed_forward.bias1 -= learning_rate * self.feed_forward.grad_bias1
        self.feed_forward.bias2 -= learning_rate * self.feed_forward.grad_bias2

        # Layer Norm 3
        self.norm3.gamma -= learning_rate * self.norm3.grad_gamma
        self.norm3.beta -= learning_rate * self.norm3.grad_beta

        

    def __call__(self, encoder_output, decoder_input):
        # Masked Multi-Head Self Attention
        masked_attention_output = self.masked_multi_attention(decoder_input, decoder_input, decoder_input)

        # Residual Connection and Normalization
        norm1_output = self.norm1(decoder_input + masked_attention_output)

        # Multi-Head Encoder-Decoder Attention
        attention_output = self.multi_attention(norm1_output, encoder_output, encoder_output)

        # Residual Connection and Normalization
        norm2_output = self.norm2(norm1_output + attention_output)

        # Feed Forward
        ff_output = self.feed_forward(norm2_output)

        # Residual Connection and Normalization
        decoder_output = self.norm3(norm2_output + ff_output)

        return decoder_output


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




class Transformer:
    def __init__(self, d_model, num_heads, d_ff, source_seq_length, target_seq_length, source_vocab_size, target_vocab_size, learning_rate = 0.01):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.source_seq_length = source_seq_length
        self.target_seq_length = target_seq_length
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.learning_rate = learning_rate

        # Input Embedding
        self.source_embedding = InputEmbedding(d_model, source_vocab_size)
        self.target_embedding = InputEmbedding(d_model, target_vocab_size)

        # Positional Encoding
        self.source_positional_encoding = PositionalEncoding(d_model, source_seq_length)
        self.target_positional_encoding = PositionalEncoding(d_model, target_seq_length)

        # Encoder Block
        self.encoder_block = EncoderBlock(d_model, num_heads, d_ff)

        # Decoder Block
        self.decoder_block = DecoderBlock(d_model, num_heads, d_ff)

        # Final Linear Layer for Output
        self.final_linear_layer = LinearLayer(d_model, target_vocab_size)


    def softmax(self, x, mask = None):
        if mask is not None:
            x = x * mask
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True)) # subtract np.max for numerical stability
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def cross_entropy_loss(self, logits, labels, mask=None):
        if mask is not None:
            logits = logits * mask
            labels = labels * mask
        logits = np.clip(logits, 1e-12, 1.0 - 1e-12)  # for numerical stability
        loss = -np.sum(labels * np.log(logits)) / len(logits)
        return loss

    def __call__(self, source_seq, target_seq):
        # Input Embedding
        source_embedded = self.source_embedding(source_seq)
        target_embedded = self.target_embedding(target_seq)

        # Positional Encoding
        source_with_position = self.source_positional_encoding(source_embedded)
        target_with_position = self.target_positional_encoding(target_embedded)

        # Encoder
        encoder_output = self.encoder_block(source_with_position)

        # Decoder
        decoder_output = self.decoder_block(encoder_output, target_with_position)

        # Final Linear Layer for Output
        output_logits = self.final_linear_layer(decoder_output)


        # Apply Softmax to get probabilities
        target_mask = (target_seq != 0)
        output_probs = self.softmax(output_logits, mask = target_mask)

        # One-hot encode the target sequence for the cross-entropy loss
        target_probs = np.eye(self.target_vocab_size)[target_seq]

        # Compute the cross-entropy loss
        loss = self.cross_entropy_loss(output_probs, target_probs, mask=target_mask)

        print(f'loss: {loss}')

        return loss
    
    def backward(self, source_seq, target_seq, loss, target_labels):
        # Compute gradient of the loss with respect to the final logits and backprop through rest of the architecture
        # if target_labels is not None:
        #     batch_size = len(source_seq)
        #     grad_output_logits = (loss - target_labels) / batch_size

        

        grad_final_linear_layer = self.final_linear_layer.backward(loss)

        grad_decoder_block = self.decoder_block.backward(grad_final_linear_layer) # will output two grads: first one for encoder block and second one for target embedding

        grad_encoder_block = self.encoder_block.backward(grad_decoder_block[0])

        grad_source_embedded = self.source_embedding.backward(source_seq, grad_encoder_block)
        grad_target_embedded = self.target_embedding.backward(target_seq, grad_decoder_block[1])

        self.source_embedding.grad_vocab_embedding = grad_source_embedded
        self.target_embedding.grad_vocab_embedding = grad_target_embedded

        self.update_parameters(self.learning_rate)

    def update_parameters(self, learning_rate):

        # Input Embedding
        self.source_embedding.vocab_embedding -= learning_rate * self.source_embedding.grad_vocab_embedding
        self.target_embedding.vocab_embedding -= learning_rate * self.target_embedding.grad_vocab_embedding

        # Encoder Block
        self.encoder_block.update_parameters(learning_rate)

        # Decoder Block
        self.decoder_block.update_parameters(learning_rate)

        # Final Linear Layer
        self.final_linear_layer.update_parameters(learning_rate)




