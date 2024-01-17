# AI 211 MACHINE PROBLEM 
# Code Authors
# [1] Jan Lendl R. Uy, 2019-00312
# [2] Ryan Roi Cayas, 2022-22085
# Section: AI 211 FZZQ

# Transformer Implementation

# Useful references:
# [1] https://www.youtube.com/watch?v=ISNdQcPhsts&t=1372s

# Import Python libraries
import numpy as np

# Import the encoder and decoder implementations
from Encoder import Encoder
from Decoder import Decoder
from Operational_Blocks import InputEmbedding, PositionalEncoding, LinearLayer, softmax

np.set_printoptions(precision=4)

# Calling the class instance is equivalent to doing a forward pass
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
        self._loss = 0.0

        # Input Embedding
        self.source_embedding = InputEmbedding(d_model, source_vocab_size)
        self.target_embedding = InputEmbedding(d_model, target_vocab_size)

        # Positional Encoding
        self.source_positional_encoding = PositionalEncoding(d_model, source_seq_length)
        self.target_positional_encoding = PositionalEncoding(d_model, target_seq_length)

        # Encoder Block
        self.encoder_block = Encoder(d_model, num_heads, d_ff)

        # Decoder Block
        self.decoder_block = Decoder(d_model, num_heads, d_ff)

        # Final Linear Layer for Output
        self.final_linear_layer = LinearLayer(d_model, target_vocab_size)
    
    def cross_entropy_loss(self, logits, labels, mask=None):
        if mask is not None:
            logits = logits * mask
            labels = labels * mask
        logits = np.clip(logits, 1e-12, 1.0 - 1e-12)  # for numerical stability

        # H(p, q) = -1/N Sum(p(x)log(q(x)))
        # p is the truth value
        # q is the predicted value
        loss = -1/len(logits) * np.sum(labels * np.log(logits))
        return loss
    
    def cross_entropy_derivative(self, logits, labels):
        # Avoid division by zero by clipping probabilities
        logits = np.clip(logits, 1e-12, 1.0 - 1e-12)
        
        # Compute the derivative of categorical cross entropy loss
        return -labels/logits
    
    def get_loss(self):

        return self._loss
        
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
        output_probs = softmax(output_logits, mask = target_mask)

        # One-hot encode the target sequence for the cross-entropy loss
        target_probs = np.eye(self.target_vocab_size)[target_seq]

        self._loss = self.cross_entropy_loss(output_probs, target_probs, target_mask)

        return output_probs

        # Compute the cross-entropy loss
        # loss = self.cross_entropy_loss(output_probs, target_probs, mask=target_mask)

        # return loss
    
    def backward(self, source_seq, target_seq, output_probs, target_labels):
        # Compute gradient of the loss with respect to the final logits and backprop through rest of the architecture
        batch_size = len(source_seq)
        grad_output_logits = self.cross_entropy_derivative(output_probs, target_labels) / batch_size

        grad_final_linear_layer = self.final_linear_layer.backward(grad_output_logits)

        # grad_decoder_block = self.decoder_block.backward(grad_final_linear_layer) # will output two grads: first one for encoder block and second one for target embedding

        # grad_encoder_block = self.encoder_block.backward(grad_decoder_block[0])

        # grad_source_embedded = self.source_embedding.backward(source_seq, grad_encoder_block)
        # grad_target_embedded = self.target_embedding.backward(target_seq, grad_decoder_block[1])

        # self.source_embedding.grad_vocab_embedding = grad_source_embedded
        # self.target_embedding.grad_vocab_embedding = grad_target_embedded

        # self.update_parameters(self.learning_rate)

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