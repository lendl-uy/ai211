from Transformer import MultiHeadAttention, LayerNorm, FeedForward

# Encoder Block
class Decoder:
    def __init__(self, d_model, num_heads, d_ff):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.d_k = d_model // num_heads

        self.masked_multi_attention = MultiHeadAttention(d_model, num_heads, masked=True)
        self.norm1 = LayerNorm(d_model)
        self.multi_attention = MultiHeadAttention(d_model, num_heads)
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