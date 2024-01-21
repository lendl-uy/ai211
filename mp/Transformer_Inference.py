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
        self.model.source_embedding_layer.vocab_embedding = loaded_params['src_vocab_embedding']
        self.model.source_pos_embedding_layer.pos_embedding = loaded_params['src_pos_embed']
        self.model.target_embedding_layer.vocab_embedding = loaded_params['tgt_vocab_embedding']
        self.model.target_pos_embedding_layer.pos_embedding = loaded_params['tgt_pos_embed']
        self.model.encoder.mha_layer.W_q = loaded_params['enc_W_q']
        self.model.encoder.mha_layer.W_k = loaded_params['enc_W_k']
        self.model.encoder.mha_layer.W_v = loaded_params['enc_W_v']
        self.model.encoder.mha_layer.W_o = loaded_params['enc_W_o']
        self.model.encoder.norm_layer_1.beta = loaded_params['enc_norm1_beta']
        self.model.encoder.norm_layer_1.gamma = loaded_params['enc_norm1_gamma']
        self.model.encoder.ff_layer.weights_1 = loaded_params['enc_ff_weights1']
        self.model.encoder.ff_layer.biases_1 = loaded_params['enc_ff_biases1']
        self.model.encoder.ff_layer.weights_2 = loaded_params['enc_ff_weights2']
        self.model.encoder.ff_layer.biases_2 = loaded_params['enc_ff_biases2']
        self.model.encoder.norm_layer_2.gamma = loaded_params['enc_norm2_gamma']
        self.model.encoder.norm_layer_2.beta = loaded_params['enc_norm2_beta']
        self.model.decoder.mmha_layer.W_q = loaded_params['dec_masked_W_q']
        self.model.decoder.mmha_layer.W_k = loaded_params['dec_masked_W_k']
        self.model.decoder.mmha_layer.W_v = loaded_params['dec_masked_W_v']
        self.model.decoder.mmha_layer.W_o = loaded_params['dec_masked_W_o']
        self.model.decoder.norm_layer_1.gamma = loaded_params['dec_norm1_gamma']
        self.model.decoder.norm_layer_1.beta = loaded_params['dec_norm1_beta']
        self.model.decoder.mha_layer.W_q = loaded_params['dec_W_q']
        self.model.decoder.mha_layer.W_k = loaded_params['dec_W_k']
        self.model.decoder.mha_layer.W_v = loaded_params['dec_W_v']
        self.model.decoder.mha_layer.W_o = loaded_params['dec_W_o']
        self.model.decoder.norm_layer_2.gamma = loaded_params['dec_norm2_gamma']
        self.model.decoder.norm_layer_2.beta = loaded_params['dec_norm2_beta']
        self.model.decoder.ff_layer.weights_1 = loaded_params['dec_ff_weights1']
        self.model.decoder.ff_layer.biases_1 = loaded_params['dec_ff_biases1']
        self.model.decoder.ff_layer.weights_2 = loaded_params['dec_ff_weights2']
        self.model.decoder.ff_layer.biases_2 = loaded_params['dec_ff_biases2']
        self.model.decoder.norm_layer_3.gamma = loaded_params['dec_norm3_gamma']
        self.model.decoder.norm_layer_3.beta = loaded_params['dec_norm3_beta']
        self.model.output_linear_layer.weights = loaded_params['final_linear_weights']
        self.model.output_linear_layer.bias = loaded_params['final_linear_bias']

        print(loaded_params['tgt_pos_embed'].shape)


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
    

    def idx_to_seq(self, idxs):
        index_to_token = self.dec_vocab_index_to_token

        sequences = [
            ' '.join([index_to_token[index] for index in row])
            for row in idxs
        ]

        return sequences



    def translate(self, src_sentence):

        print(f"Sentence: {src_sentence}")

        enc_input = self.tokenize(src_sentence)
        dec_input = np.array([self.dec_vocab_token_to_index['<SOS>']], dtype=np.int32)[np.newaxis, :]


        while dec_input[:,-1] != self.dec_vocab_token_to_index['<EOS>'] and dec_input.shape[1] <= self.max_seq_length:
            output_probs = self.model.forward(enc_input, dec_input)

            # Get the index of the class with the highest probability
            predicted_class = np.argmax(output_probs, axis=-1)

            # Append the predicted class to the decoder input
            dec_input = np.concatenate([dec_input, predicted_class[:, -1:]], axis=-1)


        # Convert the predicted sequence to tokens
        predicted_seqs = self.idx_to_seq(dec_input)

        # Join the tokens to form the translation
        translation = ' '.join(predicted_seqs)

        print(f'Translation: {translation}')


def main():
    translator = Translator()
    translator.load_model_params('mp/transformer_params.pkl')
    translator.translate('I am good.') # add your sentence here.


    



if __name__ == "__main__":
    main()