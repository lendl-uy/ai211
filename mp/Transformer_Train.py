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
    
    def one_hot_encode_targets(self, target_seq, vocab, vocab_size, seq_length):
        target_indices = vocab.seq_to_idx(target_seq, seq_length)
        target_labels = np.eye(vocab_size)[target_indices]
        return target_labels

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
        test_set = final_data[num_train:]

        # Build vocab for encoder input
        enc_vocab = self.create_vocab(train_set[:,0])
        enc_seq_length = self.max_seq_length(train_set[:,0])

        enc_vocab_size = enc_vocab.size()
        source_seq = enc_vocab.seq_to_idx(train_set[:,0], enc_seq_length) # also adds padding of 0s to reach enc_seq_length

        # Build vocab for decoder input
        dec_vocab = self.create_vocab(train_set[:,1])
        dec_seq_length = self.max_seq_length(train_set[:,1])
        
        dec_vocab_size = dec_vocab.size()
        target_seq = dec_vocab.seq_to_idx(train_set[:,1], dec_seq_length) # also adds padding of 0s to reach dec_seq_length

        # One-hot encode target sequences
        target_labels = self.one_hot_encode_targets(train_set[:, 1], dec_vocab, dec_vocab_size, dec_seq_length)

        return source_seq, target_seq, target_labels, train_set, test_set, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size



def main():
    data = DataPrep(num_sentences = 10000, train_percentage = 0.7)
    source_seq, target_seq, target_labels, train_set, test_set, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size = data('english-german-both.pkl')

    model = Transformer(d_model = 512, num_heads = 4, d_ff = 2048, 
                source_seq_length = enc_seq_length, target_seq_length = dec_seq_length, 
                source_vocab_size = enc_vocab_size, target_vocab_size = dec_vocab_size, 
                learning_rate = 0.01)
    
    num_epochs = 1
    batch_size = 32 

    for epoch in range(num_epochs):
        total_loss = 0.0

        # Iterate over batches
        for i in range(0, len(source_seq), batch_size):
            batch_source_seq = source_seq[i:i + batch_size]
            batch_target_seq = target_seq[i:i + batch_size]

            # Forward pass
            loss = model(batch_source_seq, batch_target_seq)

            target_labels = np.eye(dec_vocab_size)[batch_target_seq]

            # Backward pass: STILL NOT WORKING
            model.backward(batch_source_seq, batch_target_seq, loss, target_labels)  # Adjust target_labels as needed

            # Update parameters
            model.update_parameters()

            total_loss += loss

        average_loss = total_loss / (len(source_seq) // batch_size)
        print(f'Epoch {epoch + 1}, Average Loss: {average_loss}')


if __name__ == "__main__":
    main()