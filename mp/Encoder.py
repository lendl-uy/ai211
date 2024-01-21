from Operational_Blocks import MultiHeadAttention, Normalization, Feedforward

class Encoder:

    def __init__(self, d_model, d_ff, num_heads, dropout):

        self.mha_layer = MultiHeadAttention(d_model, num_heads, dropout) # Multi-Head Attention Layer
        self.norm_layer_1 = Normalization(d_model) # First Normalization Layer

        self.ff_layer = Feedforward(d_model, d_ff) # Feedforward Layer
        self.norm_layer_2 = Normalization(d_model) # Second Normalization Layer

    def forward(self, source_pos_embeddings, source_mask):

        mha_layer_out = self.mha_layer.forward(source_pos_embeddings, source_pos_embeddings, source_pos_embeddings, source_mask)
        norm_layer_1_out = self.norm_layer_1.forward(mha_layer_out + source_pos_embeddings)

        ff_layer_out = self.ff_layer.forward(norm_layer_1_out)
        norm_layer_2_out = self.norm_layer_2.forward(ff_layer_out + norm_layer_1_out)
        # print(f"mha_layer_out = {mha_layer_out.shape}")
        # print(f"norm_layer_1_out = {norm_layer_1_out.shape}")
        # print(f"ff_layer_out = {ff_layer_out.shape}")
        # print(f"norm_layer_2_out = {norm_layer_2_out.shape}")
        return norm_layer_2_out

    def backward(self, grad_upstream):

        grad_norm_layer_2 = self.norm_layer_2.backward(grad_upstream)
        grad_ff_layer = self.ff_layer.backward(grad_norm_layer_2)
        grad_norm_layer_1 = self.norm_layer_1.backward(grad_ff_layer + grad_norm_layer_2)
        grad_Q, grad_K, grad_V = self.mha_layer.backward(grad_norm_layer_1)

        return grad_Q + grad_K + grad_V + grad_norm_layer_1
    
    def update_parameters(self, learning_rate):

        # Multi-Head Attention
        self.mha_layer.W_q -= learning_rate * self.mha_layer.grad_W_q
        # print(f"self.mha_layer.W_q = {self.mha_layer.W_q.shape} | self.mha_layer.grad_W_q = {self.mha_layer.grad_W_q.shape}")
        self.mha_layer.W_k -= learning_rate * self.mha_layer.grad_W_k
        # print(f"self.mha_layer.W_k = {self.mha_layer.W_k.shape} | self.mha_layer.grad_W_k = {self.mha_layer.grad_W_k.shape}")
        self.mha_layer.W_v -= learning_rate * self.mha_layer.grad_W_v
        # print(f"self.mha_layer.W_v = {self.mha_layer.W_v.shape} | self.mha_layer.grad_W_v = {self.mha_layer.grad_W_v.shape}")
        self.mha_layer.W_o -= learning_rate * self.mha_layer.grad_W_o
        # print(f"self.mha_layer.W_o = {self.mha_layer.W_o.shape} | self.mha_layer.grad_W_o = {self.mha_layer.grad_W_o.shape}")

        # Layer Norm 1
        # print(f"self.norm_layer_1.gamma = {self.norm_layer_1.gamma.shape} | self.norm_layer_1.grad_gamma = {self.norm_layer_1.grad_gamma.shape}")
        self.norm_layer_1.gamma -= learning_rate * self.norm_layer_1.grad_gamma
        # print(f"self.norm_layer_1.beta = {self.norm_layer_1.beta.shape} | self.norm_layer_1.grad_beta = {self.norm_layer_1.grad_beta.shape}")
        self.norm_layer_1.beta -= learning_rate * self.norm_layer_1.grad_beta

        # Feed Forward
        # print(f"self.ff_layer.weights_1 = {self.ff_layer.weights_1.shape} | self.ff_layer.grad_weights_1 = {self.ff_layer.grad_weights_1.shape}")
        self.ff_layer.weights_1 -= learning_rate * self.ff_layer.grad_weights_1
        # print(f"self.ff_layer.weights_2 = {self.ff_layer.weights_2.shape} | self.ff_layer.grad_weights_2 = {self.ff_layer.grad_weights_2.shape}")
        self.ff_layer.weights_2 -= learning_rate * self.ff_layer.grad_weights_2
        # print(f"self.ff_layer.biases_1 = {self.ff_layer.biases_1.shape} | self.ff_layer.grad_biases_1 = {self.ff_layer.grad_biases_1.shape}")
        self.ff_layer.biases_1 -= learning_rate * self.ff_layer.grad_biases_1
        # print(f"self.ff_layer.biases_2 = {self.ff_layer.biases_2.shape} | self.ff_layer.grad_biases_2 = {self.ff_layer.grad_biases_2.shape}")
        self.ff_layer.biases_2 -= learning_rate * self.ff_layer.grad_biases_2

        # Layer Norm 2
        # print(f"self.norm_layer_2.gamma = {self.norm_layer_2.gamma.shape} | self.norm_layer_2.grad_gamma = {self.norm_layer_2.grad_gamma.shape}")
        self.norm_layer_2.gamma -= learning_rate * self.norm_layer_2.grad_gamma
        # print(f"self.norm_layer_2.gamma = {self.norm_layer_2.gamma.shape} | self.norm_layer_2.grad_gamma = {self.norm_layer_2.grad_gamma.shape}")
        self.norm_layer_2.beta -= learning_rate * self.norm_layer_2.grad_beta