from Operational_Blocks import MultiHeadAttention, Normalization, Feedforward

class Decoder:

    def __init__(self, d_model, d_ff, num_heads, dropout):

        self.mmha_layer = MultiHeadAttention(d_model, num_heads, dropout) # Masked Multi-Head Attention Layer
        self.norm_layer_1 = Normalization(d_model) # First Normalization Layer

        self.mha_layer = MultiHeadAttention(d_model, num_heads, dropout) # Multi-Head Attention Layer
        self.norm_layer_2 = Normalization(d_model) # Second Normalization Layer

        self.ff_layer = Feedforward(d_model, d_ff) # Multi-Head Attention Layer
        self.norm_layer_3 = Normalization(d_model) # Third Normalization Layer

    def forward(self, encoder_output, target_embeddings, source_mask, target_mask):

        mmha_layer_out = self.mmha_layer.forward(target_embeddings, target_embeddings, target_embeddings, target_mask)
        norm_layer_1_out = self.norm_layer_1.forward(mmha_layer_out + target_embeddings)

        mha_layer_out = self.mha_layer.forward(norm_layer_1_out, encoder_output, encoder_output, source_mask)
        norm_layer_2_out = self.norm_layer_2.forward(mha_layer_out + norm_layer_1_out)

        ff_layer_out = self.ff_layer.forward(norm_layer_2_out)
        norm_layer_3_out = self.norm_layer_3.forward(ff_layer_out + norm_layer_2_out)

        # print(f"mmha_layer_out = {mmha_layer_out.shape}")
        # print(f"norm_layer_1_out = {norm_layer_1_out.shape}")
        # print(f"mha_layer_out = {mha_layer_out.shape}")
        # print(f"norm_layer_2_out = {norm_layer_2_out.shape}")
        # print(f"ff_layer_out = {ff_layer_out.shape}")
        # print(f"norm_layer_3_out = {norm_layer_3_out.shape}")

        return norm_layer_3_out

    def backward(self, grad_upstream):

        grad_norm_layer_3 = self.norm_layer_3.backward(grad_upstream)
        grad_ff_layer = self.ff_layer.backward(grad_norm_layer_3)
        grad_norm_layer_2 = self.norm_layer_2.backward(grad_ff_layer + grad_norm_layer_3)
        grad_Q, grad_K, grad_V = self.mha_layer.backward(grad_norm_layer_2)
        grad_norm_layer_1 = self.norm_layer_1.backward(grad_Q + grad_norm_layer_2)
        grad_Q_masked, grad_K_masked, grad_V_masked = self.mmha_layer.backward(grad_norm_layer_1)

        # print(f"grad_norm_layer_3 = {grad_norm_layer_3.shape}")
        # print(f"grad_ff_layer = {grad_ff_layer.shape}")
        # print(f"grad_norm_layer_2 = {grad_norm_layer_2.shape}")
        # print(f"grad_Q = {grad_Q.shape}")
        # print(f"grad_K = {grad_K.shape}")
        # print(f"grad_V = {grad_V.shape}")
        # print(f"grad_norm_layer_1 = {grad_norm_layer_1.shape}")
        # print(f"grad_Q_masked = {grad_Q_masked.shape}")

        return grad_K_masked + grad_V_masked, grad_norm_layer_1 + grad_Q_masked + grad_K_masked + grad_V_masked
    
    def update_parameters(self, learning_rate):

        # Masked Multi-Head Attention
        # print(f"self.mmha_layer.W_q = {self.mmha_layer.W_q.shape} | self.mmha_layer.grad_W_q = {self.mmha_layer.grad_W_q.shape}")
        self.mmha_layer.W_q -= learning_rate * self.mmha_layer.grad_W_q
        # print(f"self.mmha_layer.W_k = {self.mmha_layer.W_k.shape} | self.mmha_layer.grad_W_k = {self.mmha_layer.grad_W_k.shape}")
        self.mmha_layer.W_k -= learning_rate * self.mmha_layer.grad_W_k
        # print(f"self.mmha_layer.W_v = {self.mmha_layer.W_v.shape} | self.mmha_layer.grad_W_v = {self.mmha_layer.grad_W_v.shape}")
        self.mmha_layer.W_v -= learning_rate * self.mmha_layer.grad_W_v
        # print(f"self.mmha_layer.W_o = {self.mmha_layer.W_o.shape} | self.mmha_layer.grad_W_o = {self.mmha_layer.grad_W_o.shape}")
        self.mmha_layer.W_o -= learning_rate * self.mmha_layer.grad_W_o

        # Layer Norm 1
        # print(f"self.norm_layer_1.gamma = {self.norm_layer_1.gamma.shape} | self.norm_layer_1.grad_gamma = {self.norm_layer_1.grad_gamma.shape}")
        self.norm_layer_1.gamma -= learning_rate * self.norm_layer_1.grad_gamma
        # print(f"self.norm_layer_1.beta = {self.norm_layer_1.beta.shape} | self.norm_layer_1.grad_beta = {self.norm_layer_1.grad_beta.shape}")
        self.norm_layer_1.beta -= learning_rate * self.norm_layer_1.grad_beta

        # Multi-Head Attention
        # print(f"self.mha_layer.W_q = {self.mha_layer.W_o.shape} | self.mmha_layer.grad_W_q = {self.mha_layer.grad_W_q.shape}")
        self.mha_layer.W_q -= learning_rate * self.mha_layer.grad_W_q
        # print(f"self.mha_layer.W_k = {self.mha_layer.W_k.shape} | self.mmha_layer.grad_W_k = {self.mha_layer.grad_W_k.shape}")
        self.mha_layer.W_k -= learning_rate * self.mha_layer.grad_W_k
        # print(f"self.mha_layer.W_v = {self.mha_layer.W_v.shape} | self.mmha_layer.grad_W_v = {self.mha_layer.grad_W_v.shape}")
        self.mha_layer.W_v -= learning_rate * self.mha_layer.grad_W_v
        # print(f"self.mha_layer.W_o = {self.mha_layer.W_o.shape} | self.mmha_layer.grad_W_o = {self.mha_layer.grad_W_o.shape}")
        self.mha_layer.W_o -= learning_rate * self.mha_layer.grad_W_o

        # Layer Norm 2
        # print(f"self.norm_layer_2.gamma = {self.norm_layer_1.gamma.shape} | self.norm_layer_2.grad_gamma = {self.norm_layer_1.grad_gamma.shape}")
        self.norm_layer_2.gamma -= learning_rate * self.norm_layer_2.grad_gamma
        # print(f"self.norm_layer_2.beta = {self.norm_layer_2.beta.shape} | self.norm_layer_2.grad_beta = {self.norm_layer_2.grad_beta.shape}")
        self.norm_layer_2.beta -= learning_rate * self.norm_layer_2.grad_beta

        # Feed Forward
        # print(f"self.ff_layer.weights_1 = {self.ff_layer.weights_1.shape} | self.ff_layer.grad_weights_1 = {self.ff_layer.grad_weights_1.shape}")
        self.ff_layer.weights_1 -= learning_rate * self.ff_layer.grad_weights_1
        # print(f"self.ff_layer.weights_2 = {self.ff_layer.weights_2.shape} | self.ff_layer.weights_1.grad_weights_2 = {self.ff_layer.grad_weights_2.shape}")
        self.ff_layer.weights_2 -= learning_rate * self.ff_layer.grad_weights_2
        # print(f"self.ff_layer.biases_1 = {self.ff_layer.biases_1.shape} | self.ff_layer.weights_1.grad_biases_1 = {self.ff_layer.grad_biases_1.shape}")
        self.ff_layer.biases_1 -= learning_rate * self.ff_layer.grad_biases_1
        # print(f"self.ff_layer.biases_2 = {self.ff_layer.biases_2.shape} | self.ff_layer.weights_1.grad_biases_2 = {self.ff_layer.grad_biases_2.shape}")
        self.ff_layer.biases_2 -= learning_rate * self.ff_layer.grad_biases_2

        # Layer Norm 3
        # print(f"self.norm_layer_3.gamma = {self.norm_layer_3.gamma.shape} | self.norm_layer_3.grad_gamma = {self.norm_layer_3.grad_gamma.shape}")
        self.norm_layer_3.gamma -= learning_rate * self.norm_layer_3.grad_gamma
        # print(f"self.norm_layer_3.beta = {self.norm_layer_3.beta.shape} | self.norm_layer_3.grad_beta = {self.norm_layer_3.grad_beta.shape}")
        self.norm_layer_3.beta -= learning_rate * self.norm_layer_3.grad_beta