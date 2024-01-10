# AI 211 MACHINE PROBLEM 
# Code Authors
# [1] Jan Lendl R. Uy, 2019-00312
# [2] Ryan Roi Cayas, 2023-XXXXX
# Section: AI 211 FZZQ

# Useful references:

import numpy as np
# np.set_printoptions(precision=2)



class Vocabulary:
    def __init__(self):
        self.token_to_index = {}
        self.index_to_token = {}
        self.next_index = 0

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
        # what happens if token is not in vocab?
        seqs_indices = []
        for seq in seqs:
            indices = [self.token_to_index[token] for token in seq.split() if token in self.token_to_index]

            if len(indices) > max_seq_length:
                print(f"Sequence is too long. Max sequence length is {max_seq_length}.")

            padded_indices = indices + [0] * (max_seq_length - len(indices))
            seqs_indices.append(padded_indices)
        return seqs_indices
    
    def size(self):
        return len(self.token_to_index)

    def __call__(self,train_set):
        # build vocab from train set
        for sequence in train_set[:,0]:
            tokens = sequence.split()
            for token in tokens:
                self.build_vocab(token)




class InputEmbedding:

    def __init__ (self, d_model, vocab_size):
        self.vocab = {}
        self.d_model = d_model
        self.embeding = np.random.rand()
    
    def forward_pass(self, x):
        return self.embeding * np.sqrt(self.d_model)
        



class Transformer:

    def __init__(self):
        pass
    
    # TODO: Implement the encoder architecture

    

    # TODO: Implement the decoder architecture