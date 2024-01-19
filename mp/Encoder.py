# Encoder Architecture Implementation

from Operational_Blocks import MultiHeadAttention, LayerNorm, FeedForward

class Encoder:
    def __init__(self, d_model, num_heads, d_ff, source_seq_length):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.d_k = d_model // num_heads

        self.multi_attention = MultiHeadAttention(source_seq_length, d_model, num_heads)
        self.norm_1 = LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm_2 = LayerNorm(d_model)

    def backward(self, grad_decoder_block):
        # Backward pass through the decoder block

        # Backward pass through the layer normalization after the final linear layer
        grad_norm_2 = self.norm_2.backward(grad_decoder_block)

        # Backward pass through the feedforward layer
        grad_ff = self.feed_forward.backward(grad_norm_2)

        # Reshape grad_ff to match the shape before layer normalization
        grad_ff_reshaped = grad_ff.reshape(grad_ff.shape[0], self.num_heads, self.d_k)

        # Backward pass through the layer normalization before the multi-head attention
        grad_norm1 = self.norm_1.backward(grad_ff_reshaped)

        # Backward pass through the multi-head attention
        grad_multi_attention = self.multi_attention.backward(grad_norm1)

        return grad_multi_attention[1] # pass grad key to embedding layer

    def update_parameters(self, learning_rate):
        # SGD update for each parameter in the encoder block

        # Multi-Head Attention
        self.multi_attention.W_q -= learning_rate * self.multi_attention.grad_W_q
        self.multi_attention.W_k -= learning_rate * self.multi_attention.grad_W_k
        self.multi_attention.W_v -= learning_rate * self.multi_attention.grad_W_v
        self.multi_attention.W_o -= learning_rate * self.multi_attention.grad_W_o

        # Layer Norm 1
        self.norm_1.gamma -= learning_rate * self.norm_1.grad_gamma
        self.norm_1.beta -= learning_rate * self.norm_1.grad_beta

        # Feed Forward
        self.feed_forward.weights_1 -= learning_rate * self.feed_forward.grad_weights_1
        self.feed_forward.weights_2 -= learning_rate * self.feed_forward.grad_weights_2
        self.feed_forward.biases_1 -= learning_rate * self.feed_forward.grad_biases_1
        self.feed_forward.biases_2 -= learning_rate * self.feed_forward.grad_biases_2

        # Layer Norm 2
        self.norm_2.gamma -= learning_rate * self.norm_2.grad_gamma
        self.norm_2.beta -= learning_rate * self.norm_2.grad_beta

    def __call__(self, input):
        # Multi-Head Self Attention
        attention_output = self.multi_attention(input, input, input)
        print(f"enc attention_output = {attention_output.shape}")
        # Residual Connection and Normalization
        norm1_output = self.norm_1(input + attention_output)
        print(f"enc norm1_output = {norm1_output.shape}")
        # Feed Forward
        ff_output = self.feed_forward(norm1_output)
        print(f"enc ff_output = {ff_output.shape}")
        # Residual Connection and Normalization
        encoder_output = self.norm_2(norm1_output + ff_output)
        print(f"enc encoder_output = {encoder_output.shape}")
        return encoder_output