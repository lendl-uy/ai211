# AI 211 MACHINE PROBLEM 
# Code Authors
# [1] Jan Lendl R. Uy, 2019-00312
# [2] Ryan Roi Cayas, 2023-XXXXX
# Section: AI 211 FZZQ

# Training the Transformer

import os
import pickle
import numpy as np
from Transformer import *


class DataPrep:
    def __init__(self, num_sentences, train_percentage):
        self.num_sentences = num_sentences
        self.train_percentage = train_percentage
    
    def create_vocab(self, data):
        vocab = Vocabulary()

        for seq in data:
            tokens = seq.split()
            for token in tokens:
                vocab.build_vocab(token)
        return vocab
    
    def max_seq_length(self,data):
        return max(len(seq.split()) for seq in data)

    def __call__(self, file):
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        
        dataset = pickle.load(open(os.path.join(__location__, file), 'rb'))
        np.random.shuffle(dataset)
        final_data = dataset[:self.num_sentences,:]

        # Append <SOS> and <EOS> tags to start and end of sentences resp.
        for i in range(final_data[:,0].size):
            final_data[i, 0] = "<SOS> " + final_data[i, 0] + " <EOS>"
            final_data[i, 1] = "<SOS> " + final_data[i, 1] + " <EOS>"

        num_train = int(self.num_sentences * self.train_percentage)
        train_set = final_data[:num_train]

        # Build vocab for encoder input
        enc_vocab = self.create_vocab(train_set[:,0])
        enc_seq_length = self.max_seq_length(train_set[:,0])

        enc_vocab_size = enc_vocab.size()
        train_enc = enc_vocab.seq_to_idx(train_set[:,0], enc_seq_length) # also adds padding of 0s to reach enc_seq_length

        # Build vocab for decoder input
        dec_vocab = self.create_vocab(train_set[:,1])
        dec_seq_length = self.max_seq_length(train_set[:,1])
        
        dec_vocab_size = dec_vocab.size()
        train_dec = dec_vocab.seq_to_idx(train_set[:,1], dec_seq_length) # also adds padding of 0s to reach dec_seq_length

        return train_enc, train_dec, train_set, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size


def main():
    data = DataPrep(num_sentences = 10000, train_percentage = 0.7)
    train_enc, train_dec, train_set, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size = data('english-german-both.pkl')

if __name__ == "__main__":
    main()