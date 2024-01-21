import numpy as np

from Transformer_Constants import *
from Operational_Blocks import (InputEmbedding, 
                                PositionalEncoding, 
                                Linear, 
                                Softmax)
from Encoder import Encoder
from Decoder import Decoder

class Transformer:

    def __init__(self, d_model, num_heads, d_ff, source_seq_length, target_seq_length, 
                source_vocab_size, target_vocab_size, learning_rate):
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.source_seq_length = source_seq_length
        self.target_seq_length = target_seq_length
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.learning_rate = learning_rate

        # Input Embedding Block
        self.source_embedding_layer = InputEmbedding(d_model, source_vocab_size)
        self.target_embedding_layer = InputEmbedding(d_model, target_vocab_size)

        # Positional Encoding Block
        self.source_pos_embedding_layer = PositionalEncoding(d_model, source_seq_length)
        self.target_pos_embedding_layer = PositionalEncoding(d_model, target_seq_length)

        # Encoder Block
        self.encoder = Encoder(D_MODEL, D_FF, HEADS, DROPOUT_PERCENT)

        # Decoder Block
        self.decoder = Decoder(D_MODEL, D_FF, HEADS, DROPOUT_PERCENT)

        # Output Blocks
        self.output_linear_layer = Linear(D_MODEL, target_vocab_size)
        self.output_softmax_layer = Softmax()

    def forward(self, source_seq, target_seq):

        # Create input embeddings 
        source = self.source_embedding_layer(source_seq)
        target = self.target_embedding_layer(target_seq)

        # Apply positional encodings to input embeddings
        source_pos_encoding = self.source_pos_embedding_layer.forward(source)
        target_pos_encoding = self.target_pos_embedding_layer.forward(target)

        # Forward pass through the encoder
        encoder_output = self.encoder.forward(source_pos_encoding)
        decoder_output = self.decoder.forward(encoder_output, target_pos_encoding)

        # Forward pass through the output blocks
        logits = self.output_linear_layer.forward(decoder_output)
        predictions = self.output_softmax_layer.forward(logits)

        # print(f"logits = {logits.shape}")
        # print(f"predictions = {predictions.shape}")

        return predictions

    def backward(self, grad_upstream, source_seq, target_seq):

        grad_output_softmax = self.output_softmax_layer.backward(grad_upstream)
        grad_output_linear_layer = self.output_linear_layer.backward(grad_output_softmax)

        grad_encoder_upstream, grad_decoder = self.decoder.backward(grad_output_linear_layer)
        grad_encoder = self.encoder.backward(grad_encoder_upstream)

        grad_source_embedded = self.source_embedding_layer.backward(source_seq, grad_encoder)
        grad_target_embedded = self.target_embedding_layer.backward(target_seq, grad_decoder)

        self.source_embedding_layer.grad_vocab_embedding = grad_source_embedded
        self.target_embedding_layer.grad_vocab_embedding = grad_target_embedded

        self.update_parameters()

    def update_parameters(self):

        # Input Embedding
        self.source_embedding_layer.vocab_embedding -= self.learning_rate * self.source_embedding_layer.grad_vocab_embedding
        self.target_embedding_layer.vocab_embedding -= self.learning_rate * self.target_embedding_layer.grad_vocab_embedding

        # Encoder Block
        self.encoder.update_parameters(self.learning_rate)

        # Decoder Block
        self.decoder.update_parameters(self.learning_rate)

        # Final Linear Layer
        self.output_linear_layer.update_parameters(self.learning_rate)