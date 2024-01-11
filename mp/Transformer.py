# AI 211 MACHINE PROBLEM 
# Code Authors
# [1] Jan Lendl R. Uy, 2019-00312
# [2] Ryan Roi Cayas, 2023-XXXXX
# Section: AI 211 FZZQ

# Useful references:

import numpy as np
np.set_printoptions(precision=4)



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
        self.vocab_embedding = np.random.rand(vocab_size,d_model)

    def embedding(self, input_indices):
        input_indices = np.array(input_indices)
        embedded_output = self.vocab_embedding[input_indices]
        return embedded_output
    
    def __call__(self, input_indices):
        input_indices = np.array(input_indices)
        return self.embedding(input_indices) * np.sqrt(self.d_model)
    

class PositionalEncoding:

    def __init__(self, d_model, seq_length): # no dropout yet
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
        
        print("Pos Embedding: \n")
        print(pos_embed_3d)

        input_embeddings += pos_embed_3d
        return input_embeddings



        



class Transformer:

    def __init__(self):
        pass
    
    # TODO: Implement the encoder architecture

    

    # TODO: Implement the decoder architecture