# AI 211 MACHINE PROBLEM 
# Code Authors
# [1] Jan Lendl R. Uy, 2019-00312
# [2] Ryan Roi Cayas, 2022-22085
# Section: AI 211 FZZQ

# Transformer Implementation Tests

from Transformer_Constants import *
from Operational_Blocks import *
from Transformer_Train import *

dataset_path = "english-german-both.pkl"
    
def construct_vocabulary():
    data = DataPrep(num_sentences = SENTENCE_LENGTH, train_percentage = TRAIN_PERCENTAGE)
    source_seq, target_seq, target_labels, train_set, test_set, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size = data(dataset_path)
    print("train set:")
    print(train_set)
    print()

    print("train_enc:")
    print(source_seq)
    print(f"enc_seq_length: {enc_seq_length}, enc_vocab_size: {enc_vocab_size}")
    print()

    print("train_dec:")
    print(target_labels)
    print(f"dec_seq_length: {dec_seq_length}, dec_vocab_size: {dec_vocab_size}")

def construct_input_embeddings():
    data = DataPrep(num_sentences = SENTENCE_LENGTH, train_percentage = TRAIN_PERCENTAGE)
    source_seq, target_seq, target_labels, train_set, test_set, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size = data(dataset_path)

    print("train_enc:")
    print(source_seq)
    print()

    src_embed = InputEmbedding(D_MODEL, enc_vocab_size)
    print("Input Embedding:\n")
    print(src_embed.embedding(source_seq))
    print("Shape: ",src_embed.embedding(source_seq).shape)

def construct_positional_encodings():
    data = DataPrep(num_sentences = SENTENCE_LENGTH, train_percentage = TRAIN_PERCENTAGE)
    source_seq, target_seq, target_labels, train_set, test_set, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size = data(dataset_path)

    print("train_enc:")
    print(source_seq)
    print()

    print("Enc_seq_length: ",enc_seq_length)

    d_model = 10
    src_embed = InputEmbedding(D_MODEL, enc_vocab_size)
    input_embed = src_embed.embedding(source_seq)
    print("\nInput Embedding:\n")
    print(input_embed)

    src_pos = PositionalEncoding(D_MODEL, enc_seq_length)
    final_embed = src_pos(input_embed)
    print("\nFinal Embedding:\n",final_embed)
    print("Final Embedding Shape: ",final_embed.shape)

def construct_multi_head_attention_matrix():
    data = DataPrep(num_sentences = SENTENCE_LENGTH, train_percentage = TRAIN_PERCENTAGE)
    source_seq, target_seq, target_labels, train_set, test_set, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size = data(dataset_path)

    d_model = 12
    src_embed = InputEmbedding(D_MODEL, enc_vocab_size)
    input_embed = src_embed.embedding(source_seq)

    src_pos = PositionalEncoding(D_MODEL, enc_seq_length)
    final_embed = src_pos(input_embed)

    num_heads = 4
    src_multi_attention = MultiHeadAttention(d_model, num_heads, masked = True)
    attention_out  = src_multi_attention(final_embed, final_embed, final_embed)

    print("Attention Out Shape: ", attention_out.shape)

def perform_layer_normalization():
    data = DataPrep(num_sentences = SENTENCE_LENGTH, train_percentage = TRAIN_PERCENTAGE)
    source_seq, target_seq, target_labels, train_set, test_set, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size = data(dataset_path)

    d_model = 12
    src_embed = InputEmbedding(D_MODEL, enc_vocab_size)
    input_embed = src_embed.embedding(source_seq)

    src_pos = PositionalEncoding(D_MODEL, enc_seq_length)
    final_embed = src_pos(input_embed)

    num_heads = 4
    src_multi_attention = MultiHeadAttention(d_model, num_heads, masked = True)
    attention_out  = src_multi_attention(final_embed, final_embed, final_embed)

    src_layer_norm = LayerNorm(d_model)
    normalized_out = src_layer_norm(attention_out)

    print("Normalized Out Shape: ", normalized_out.shape)

def construct_encoder():
    data = DataPrep(num_sentences = SENTENCE_LENGTH, train_percentage = TRAIN_PERCENTAGE)
    source_seq, target_seq, target_labels, train_set, test_set, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size = data(dataset_path)

    src_embed = InputEmbedding(D_MODEL ,enc_vocab_size)
    input_embed = src_embed.embedding(source_seq)

    src_pos = PositionalEncoding(D_MODEL, enc_seq_length)
    final_embed = src_pos(input_embed)

    src_multi_attention = MultiHeadAttention(D_MODEL, HEADS, masked = True)
    attention_out  = src_multi_attention(final_embed, final_embed, final_embed)

    src_layer_norm = LayerNorm(D_MODEL)
    normalized_out = src_layer_norm(attention_out)

    src_nn = FeedForward(D_MODEL, D_FF)
    NN_out = src_nn(normalized_out)

    print("NN Out Shape: ", NN_out.shape)

def main():
    # construct_vocabulary()
    # construct_input_embeddings()
    # construct_positional_encodings()
    construct_multi_head_attention_matrix()
    # perform_layer_normalization()
    # construct_encoder()

if __name__ == "__main__":
    main()