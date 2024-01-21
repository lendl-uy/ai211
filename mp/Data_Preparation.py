import numpy as np
import os
import pickle

from Transformer_Constants import *

class Vocabulary:
    def __init__(self):
        self.token_to_index = {}
        self.index_to_token = {}
        self.next_index = 2

        # add <PAD> and <UNK> to vocab
        self.token_to_index['<PAD>'] = 0
        self.index_to_token[0] = '<PAD>'
        self.token_to_index['<UNK>'] = 1
        self.index_to_token[1] = '<UNK>'

    def build_vocab(self, token):
        if token not in self.token_to_index:
            index = self.next_index
            self.token_to_index[token] = index
            self.index_to_token[index] = token
            self.next_index += 1


    def seq_to_idx(self, seqs, max_seq_length):
        seqs_indices = []
        for seq in seqs:
            indices = [self.token_to_index[token] for token in seq.split() if token in self.token_to_index]

            if len(indices) > max_seq_length:
                print(f"Sequence is too long. Max sequence length is {max_seq_length}.")

            padded_indices = indices + [0] * (max_seq_length - len(indices))
            seqs_indices.append(padded_indices)
        return seqs_indices
    
    def idx_to_seq(self, idxs):
        index_to_token = self.index_to_token

        sequences = [
            ' '.join([index_to_token[index] for index in row])
            for row in idxs
        ]

        return sequences
    
    def size(self):
        return len(self.token_to_index) + 1 # we add 1 for the padding

    def __call__(self,train_set):
        # build vocab from train set
        for sequence in train_set[:,0]:
            tokens = sequence.split()
            for token in tokens:
                self.build_vocab(token)

    def get_vocab(self):
        return self.token_to_index

class DataPreparation:
    def __init__(self, num_sentences, train_percentage):
        self.num_sentences = num_sentences
        self.train_percentage = train_percentage
        self.enc_vocab = None
        self.dec_voacb = None
    
    def create_vocab(self, data):
        vocab = Vocabulary()

        for seq in data:
            tokens = seq.split()
            for token in tokens:
                vocab.build_vocab(token)
        return vocab
    
    def max_seq_length(self,data):
        return max(len(seq.split()) for seq in data)
    
    def one_hot_encode_targets(self, target_seq, vocab, vocab_size, seq_length):
        target_indices = vocab.seq_to_idx(target_seq, seq_length)
        target_labels = np.eye(vocab_size)[target_indices]
        return target_labels

    def __call__(self, file):
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        
        dataset = pickle.load(open(os.path.join(__location__, file), 'rb'))
        np.random.shuffle(dataset)

        # Column 1 contains encoder input data
        # Column 2 contains decoder input data
        final_data = dataset[:self.num_sentences,:]

        # Append <SOS> and <EOS> tags to start and end of sentences resp.
        for i in range(final_data[:,0].size):
            final_data[i, 0] = "<SOS> " + final_data[i, 0] + " <EOS>"
            final_data[i, 1] = "<SOS> " + final_data[i, 1] + " <EOS>"

        num_train = int(self.num_sentences * self.train_percentage)
        train_set = final_data[:num_train]
        test_set = final_data[num_train:]

        # Build vocab for encoder input
        self.enc_vocab = self.create_vocab(train_set[:,0])
        # enc_seq_length = self.max_seq_length(train_set[:,0])
        enc_vocab_size = self.enc_vocab.size()
        source_seq = self.enc_vocab.seq_to_idx(train_set[:,0], MAX_SEQ_LENGTH) # also adds padding of 0s to reach enc_seq_length

        # Build vocab for decoder input
        self.dec_vocab = self.create_vocab(train_set[:,1])
        # dec_seq_length = self.max_seq_length(train_set[:,1])
        
        dec_vocab_size = self.dec_vocab.size()
        target_seq = self.dec_vocab.seq_to_idx(train_set[:,1], MAX_SEQ_LENGTH) # also adds padding of 0s to reach dec_seq_length

        # One-hot encode target sequences
        target_labels = self.one_hot_encode_targets(train_set[:, 1], self.dec_vocab, dec_vocab_size, MAX_SEQ_LENGTH)

        return source_seq, target_seq, target_labels, train_set, test_set, enc_vocab_size, dec_vocab_size, self.enc_vocab, self.dec_vocab