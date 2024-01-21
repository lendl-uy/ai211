# AI 211 MACHINE PROBLEM 
# Code Authors
# [1] Jan Lendl R. Uy, 2019-00312
# [2] Ryan Roi Cayas, 2022-22085
# Section: AI 211 FZZQ

# Performing Inference on Trained Transformer

# Reference: https://github.com/ajhalthor/Transformer-Neural-Network/blob/main/Transformer_Trainer_Notebook.ipynb


import numpy as np
import pickle
import string
from Transformer import Transformer


class Translator: 
    def load_model_params(self, filepath):
        with open(filepath, 'rb') as f:
            loaded_params = pickle.load(f)

        self.model = Transformer(d_model = loaded_params['d_model'], num_heads = loaded_params['heads'], d_ff = loaded_params['d_ff'], 
                            source_seq_length = loaded_params['max_seq_length'], target_seq_length = loaded_params['max_seq_length'], 
                            source_vocab_size = loaded_params['src_vocab_embedding'].shape[0], target_vocab_size = loaded_params['tgt_vocab_embedding'].shape[0], 
                            learning_rate = loaded_params['learning_rate'])
        

        self.max_seq_length = loaded_params['max_seq_length']
        self.enc_vocab_token_to_index = loaded_params['enc_vocab_token_to_index']
        self.enc_vocab_index_to_token = loaded_params['enc_vocab_index_to_token']
        self.dec_vocab_token_to_index = loaded_params['dec_vocab_token_to_index']
        self.dec_vocab_index_to_token = loaded_params['dec_vocab_index_to_token']
        self.model.source_embedding.vocab_embedding = loaded_params['src_vocab_embedding']
        self.model.source_positional_encoding.pos_embed = loaded_params['src_pos_embed']
        self.model.target_embedding.vocab_embedding = loaded_params['tgt_vocab_embedding']
        self.model.target_positional_encoding.pos_embed = loaded_params['tgt_pos_embed']
        self.model.encoder_block.multi_attention.W_q = loaded_params['enc_W_q']
        self.model.encoder_block.multi_attention.W_k = loaded_params['enc_W_k']
        self.model.encoder_block.multi_attention.W_v = loaded_params['enc_W_v']
        self.model.encoder_block.multi_attention.W_o = loaded_params['enc_W_o']
        self.model.encoder_block.norm_1.beta = loaded_params['enc_norm1_beta']
        self.model.encoder_block.norm_1.gamma = loaded_params['enc_norm1_gamma']
        self.model.encoder_block.feed_forward.weights_1 = loaded_params['enc_ff_weights1']
        self.model.encoder_block.feed_forward.biases_1 = loaded_params['enc_ff_biases1']
        self.model.encoder_block.feed_forward.weights_2 = loaded_params['enc_ff_weights2']
        self.model.encoder_block.feed_forward.biases_2 = loaded_params['enc_ff_biases2']
        self.model.encoder_block.norm_2.gamma = loaded_params['enc_norm2_gamma']
        self.model.encoder_block.norm_2.beta = loaded_params['enc_norm2_beta']
        self.model.decoder_block.masked_multi_attention.W_q = loaded_params['dec_masked_W_q']
        self.model.decoder_block.masked_multi_attention.W_k = loaded_params['dec_masked_W_k']
        self.model.decoder_block.masked_multi_attention.W_v = loaded_params['dec_masked_W_v']
        self.model.decoder_block.masked_multi_attention.W_o = loaded_params['dec_masked_W_o']
        self.model.decoder_block.norm_1.gamma = loaded_params['dec_norm1_gamma']
        self.model.decoder_block.norm_1.beta = loaded_params['dec_norm1_beta']
        self.model.decoder_block.multi_attention.W_q = loaded_params['dec_W_q']
        self.model.decoder_block.multi_attention.W_k = loaded_params['dec_W_k']
        self.model.decoder_block.multi_attention.W_v = loaded_params['dec_W_v']
        self.model.decoder_block.multi_attention.W_o = loaded_params['dec_W_o']
        self.model.decoder_block.norm_2.gamma = loaded_params['dec_norm2_gamma']
        self.model.decoder_block.norm_2.beta = loaded_params['dec_norm2_beta']
        self.model.decoder_block.feed_forward.weights_1 = loaded_params['dec_ff_weights1']
        self.model.decoder_block.feed_forward.biases_1 = loaded_params['dec_ff_biases1']
        self.model.decoder_block.feed_forward.weights_2 = loaded_params['dec_ff_weights2']
        self.model.decoder_block.feed_forward.biases_2 = loaded_params['dec_ff_biases2']
        self.model.decoder_block.norm_3.gamma = loaded_params['dec_norm3_gamma']
        self.model.decoder_block.norm_3.beta = loaded_params['dec_norm3_beta']
        self.model.final_linear_layer.weights = loaded_params['final_linear_weights']
        self.model.final_linear_layer.bias = loaded_params['final_linear_bias']           


    def tokenize (self, src_sentence):
        # Remove punctuations from the sentence
        src_sentence = src_sentence.translate(str.maketrans("", "", string.punctuation))

        # tokenize sentence
        tokens = src_sentence.lower().split()
        tokens = ['<SOS>'] + tokens + ['<EOS>']

        for token in tokens:
            if self.enc_vocab_token_to_index.get(token) == None:
                raise ValueError(f'Word is not in vocabulary: {token}')
            
        if len(tokens) > self.max_seq_length:
            raise ValueError(f'Sentence exceeds maximum sequence length of {self.max_seq_length}')
        
        # Convert tokens to indices
        padded_tokens = tokens[:self.max_seq_length] + ['<PAD>'] * (self.max_seq_length - len(tokens))
        indices = np.array([self.enc_vocab_token_to_index[token] for token in padded_tokens], dtype=np.int32)

        if indices.ndim == 1:
            indices = indices[np.newaxis, :]

        return indices



    def translate(self, src_sentence):

        enc_input = self.tokenize(src_sentence)
        dec_input = np.array([self.dec_vocab_token_to_index['<SOS>']], dtype=np.int32)[np.newaxis, :]

        output_probs = self.model(enc_input, dec_input)

        print(output_probs)

        # while dec_input[-1] != self.dec_vocab_token_to_index['<EOS>'] and len(dec_input) <= self.max_seq_length:

        #     output_probs = self.model(enc_input, dec_input)

        # # Get the index of the class with the highest probability
        # predicted_class = np.argmax(output_probs, axis=-1)
        # print(f'Predicted Indices: \n{predicted_class}\n')
        # print(f'Predicted Sequences:')
        # for seq in predicted_seqs:
        #     print(seq)


def main():
    translator = Translator()
    translator.load_model_params('transformer_params.pkl')
    translator.translate('My friend.')


    


    







if __name__ == "__main__":
    main()