# AI 211 MACHINE PROBLEM 
# Code Authors
# [1] Jan Lendl R. Uy, 2019-00312
# [2] Ryan Roi Cayas, 2022-22085
# Section: AI 211 FZZQ

# Training the Transformer

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time


from Transformer_Constants import *
from Transformer import Transformer

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

    # def get_index(self, token):
    #     return self.token_to_index.get(token, None)

    # def get_token(self, index):
    #     return self.index_to_token.get(index, None)

    def seq_to_idx(self, seqs, max_seq_length):
        seqs_indices = []
        for seq in seqs:
            indices = [self.token_to_index.get(token, self.token_to_index['<UNK>']) for token in seq.split()]

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
        return len(self.token_to_index)

    def __call__(self,train_set):
        # build vocab from train set
        for sequence in train_set[:,0]:
            tokens = sequence.split()
            for token in tokens:
                self.build_vocab(token)

class DataPreparation:
    def __init__(self, num_sentences, train_percentage, max_seq_length):
        self.num_sentences = num_sentences
        self.train_percentage = train_percentage
        self.max_seq_length = max_seq_length
    
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
        enc_vocab = self.create_vocab(train_set[:,0])
        # enc_seq_length = self.max_seq_length(train_set[:,0])
        enc_seq_length = self.max_seq_length

        enc_vocab_size = enc_vocab.size()
        source_seq = enc_vocab.seq_to_idx(train_set[:,0], enc_seq_length) # also adds padding of 0s to reach enc_seq_length

        # Build vocab for decoder input
        dec_vocab = self.create_vocab(train_set[:,1])
        # dec_seq_length = self.max_seq_length(train_set[:,1])
        dec_seq_length = self.max_seq_length
        
        dec_vocab_size = dec_vocab.size()
        target_seq = dec_vocab.seq_to_idx(train_set[:,1], dec_seq_length) # also adds padding of 0s to reach dec_seq_length

        # One-hot encode target sequences
        target_labels = self.one_hot_encode_targets(train_set[:, 1], dec_vocab, dec_vocab_size, dec_seq_length)

        return source_seq, target_seq, target_labels, train_set, test_set, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size, enc_vocab, dec_vocab

def main():
    data = DataPreparation(num_sentences = SENTENCE_LENGTH, train_percentage = TRAIN_PERCENTAGE, max_seq_length = MAX_SEQ_LENGTH)
    source_seq, target_seq, target_labels, train_set, test_set, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size, enc_vocab, dec_vocab = data(DATASET_PATH)
    
    model = Transformer(d_model = D_MODEL, num_heads = HEADS, d_ff = D_FF, 
                        source_seq_length = MAX_SEQ_LENGTH, target_seq_length = MAX_SEQ_LENGTH, 
                        source_vocab_size = enc_vocab_size, target_vocab_size = dec_vocab_size, 
                        learning_rate = LEARNING_RATE)
    
    train_losses = []
    test_losses = []

    print(train_set)

    # For testing
    test_source_seq = enc_vocab.seq_to_idx(test_set[:,0], enc_seq_length)
    test_target_seq = dec_vocab.seq_to_idx(test_set[:,1], dec_seq_length)

    for epoch in range(EPOCHS):
        start_time = time.time()

        total_loss = 0.0

        # Iterate over batches
        for i in range(0, len(source_seq), BATCH_SIZE):
            batch_source_seq = source_seq[i:i + BATCH_SIZE]
            batch_target_seq = target_seq[i:i + BATCH_SIZE]

            # Forward pass
            model_output = model(batch_source_seq, batch_target_seq)

            target_labels = np.eye(dec_vocab_size)[batch_target_seq]

            # print(f"target_labels = {target_labels}")

            # Back propagate errors and update parameters
            model.backward(batch_source_seq, batch_target_seq, model_output, target_labels)  # Adjust target_labels as needed

            total_loss += model.get_loss()
            
            print(f"Done Batch {(i // BATCH_SIZE) + 1}")

        average_loss = total_loss / (len(source_seq) // BATCH_SIZE)
        train_losses.append(average_loss)

        # compute test loss
        model_output = model(test_source_seq, test_target_seq, eval = True)
        test_loss = model.get_loss(eval=True)
        test_losses.append(test_loss)

        print(f'Epoch {epoch + 1}, Average Train Loss: {average_loss}, Test Loss: {test_loss}')
        print(f'Epoch run time: {(time.time() - start_time)}')

    # Plotting the losses
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss Over Epochs')
    plt.legend()
    plt.show()

    # Save model parameters
    transformer_params = {
    "train_losses": train_losses,
    "test_losses": test_losses,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "train_percentage": TRAIN_PERCENTAGE,
    "sentence_length": SENTENCE_LENGTH,
    "max_seq_length": MAX_SEQ_LENGTH,
    "learning_rate": LEARNING_RATE,
    "d_model": D_MODEL,
    "d_ff": D_FF,
    "heads": HEADS,
    "threshold": THRESHOLD,
    "enc_vocab_token_to_index": enc_vocab.token_to_index,
    "enc_vocab_index_to_token": enc_vocab.index_to_token,
    "dec_vocab_token_to_index": dec_vocab.token_to_index,
    "dec_vocab_index_to_token": dec_vocab.index_to_token,
    "src_vocab_embedding": model.source_embedding.vocab_embedding,
    "src_pos_embed": model.source_positional_encoding.pos_embed, 
    "tgt_vocab_embedding": model.target_embedding.vocab_embedding,
    "tgt_pos_embed": model.target_positional_encoding.pos_embed,
    "enc_W_q": model.encoder_block.multi_attention.W_q,
    "enc_W_k": model.encoder_block.multi_attention.W_k,
    "enc_W_v": model.encoder_block.multi_attention.W_v,
    "enc_W_o": model.encoder_block.multi_attention.W_o,
    "enc_norm1_beta": model.encoder_block.norm_1.beta,
    "enc_norm1_gamma": model.encoder_block.norm_1.gamma,
    "enc_ff_weights1": model.encoder_block.feed_forward.weights_1,
    "enc_ff_biases1": model.encoder_block.feed_forward.biases_1,
    "enc_ff_weights2": model.encoder_block.feed_forward.weights_2,
    "enc_ff_biases2": model.encoder_block.feed_forward.biases_2,
    "enc_norm2_gamma": model.encoder_block.norm_2.gamma,
    "enc_norm2_beta": model.encoder_block.norm_2.beta,
    "dec_masked_W_q": model.decoder_block.masked_multi_attention.W_q,
    "dec_masked_W_k": model.decoder_block.masked_multi_attention.W_k,
    "dec_masked_W_v": model.decoder_block.masked_multi_attention.W_v,
    "dec_masked_W_o": model.decoder_block.masked_multi_attention.W_o,
    "dec_norm1_gamma": model.decoder_block.norm_1.gamma,
    "dec_norm1_beta": model.decoder_block.norm_1.beta,
    "dec_W_q": model.decoder_block.multi_attention.W_q,
    "dec_W_k": model.decoder_block.multi_attention.W_k,
    "dec_W_v": model.decoder_block.multi_attention.W_v,
    "dec_W_o": model.decoder_block.multi_attention.W_o,
    "dec_norm2_gamma": model.decoder_block.norm_2.gamma,
    "dec_norm2_beta": model.decoder_block.norm_2.beta,
    "dec_ff_weights1": model.decoder_block.feed_forward.weights_1,
    "dec_ff_biases1": model.decoder_block.feed_forward.biases_1,
    "dec_ff_weights2": model.decoder_block.feed_forward.weights_2,
    "dec_ff_biases2": model.decoder_block.feed_forward.biases_2,
    "dec_norm3_gamma": model.decoder_block.norm_3.gamma,
    "dec_norm3_beta": model.decoder_block.norm_3.beta,
    "final_linear_weights": model.final_linear_layer.weights,
    "final_linear_bias": model.final_linear_layer.bias
    }

    with open('transformer_params.pkl', 'wb') as file:
        pickle.dump(transformer_params, file)


    



if __name__ == "__main__":
    main()