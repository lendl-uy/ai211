# AI 211 MACHINE PROBLEM 
# Code Authors
# [1] Jan Lendl R. Uy, 2019-00312
# [2] Ryan Roi Cayas, 2022-22085
# Section: AI 211 FZZQ

# Transformer Implementation Tests

from Transformer_Constants import *
from Operational_Blocks import *
from Transformer_Train import *

def construct_vocabulary():
    data = DataPreparation(num_sentences = SENTENCE_LENGTH, train_percentage = TRAIN_PERCENTAGE)
    source_seq, target_seq, target_labels, train_set, test_set, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size = data(DATASET_PATH)
    print(f"train set: \n{train_set}\n")

    print(f"train_enc: \n{source_seq}\n")
    print(f"enc_seq_length: {enc_seq_length}, enc_vocab_size: {enc_vocab_size}\n")

    print(f"train_dec: \n{target_seq}\n")
    print(f"dec_seq_length: {dec_seq_length}, dec_vocab_size: {dec_vocab_size}")

def construct_input_embeddings():
    data = DataPreparation(num_sentences = SENTENCE_LENGTH, train_percentage = TRAIN_PERCENTAGE)
    source_seq, target_seq, target_labels, train_set, test_set, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size = data(DATASET_PATH)

    print(f"train_enc: \n{source_seq}\n")

    src_embed = InputEmbedding(D_MODEL, enc_vocab_size)
    print(f"Input Embedding: \n{src_embed.embedding(source_seq)}\n")
    print("Shape: ",src_embed.embedding(source_seq).shape)

def construct_positional_encodings():
    data = DataPreparation(num_sentences = SENTENCE_LENGTH, train_percentage = TRAIN_PERCENTAGE)
    source_seq, target_seq, target_labels, train_set, test_set, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size = data(DATASET_PATH)

    print(f"train_enc: \n{source_seq}\n")

    print(f"Enc_seq_length: {enc_seq_length}\n")

    # d_model = 10
    src_embed = InputEmbedding(D_MODEL, enc_vocab_size)
    input_embed = src_embed.embedding(source_seq)
    print(f"\n Input Embedding:\n {input_embed}\n")

    src_pos = PositionalEncoding(D_MODEL, enc_seq_length)
    final_embed = src_pos(input_embed)
    print(f"\n Final Embedding:\n {final_embed}")
    print(f"Final Embedding Shape: {final_embed.shape}")

def construct_multi_head_attention_matrix():
    data = DataPreparation(num_sentences = SENTENCE_LENGTH, train_percentage = TRAIN_PERCENTAGE)
    source_seq, target_seq, target_labels, train_set, test_set, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size = data(DATASET_PATH)

    # d_model = 12
    src_embed = InputEmbedding(D_MODEL, enc_vocab_size)
    input_embed = src_embed.embedding(source_seq)

    src_pos = PositionalEncoding(D_MODEL, enc_seq_length)
    final_embed = src_pos(input_embed)

    # num_heads = 4
    src_multi_attention = MultiHeadAttention(D_MODEL, HEADS, masked = True)
    attention_out = src_multi_attention(final_embed, final_embed, final_embed)

    print(f"attention_out: {attention_out}\n")
    print(f"Attention Out Shape: {attention_out.shape}")

def perform_layer_normalization():
    data = DataPreparation(num_sentences = SENTENCE_LENGTH, train_percentage = TRAIN_PERCENTAGE)
    source_seq, target_seq, target_labels, train_set, test_set, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size = data(DATASET_PATH)

    # d_model = 12
    src_embed = InputEmbedding(D_MODEL, enc_vocab_size)
    input_embed = src_embed.embedding(source_seq)

    src_pos = PositionalEncoding(D_MODEL, enc_seq_length)
    final_embed = src_pos(input_embed)

    # num_heads = 4
    src_multi_attention = MultiHeadAttention(D_MODEL, HEADS, masked = True)
    attention_out  = src_multi_attention(final_embed, final_embed, final_embed)

    src_layer_norm = LayerNorm(D_MODEL)
    normalized_out = src_layer_norm(attention_out)

    print(f"Normalized Out: {normalized_out}")
    print(f"Normalized Out Shape: {normalized_out.shape}")

def construct_encoder():
    data = DataPreparation(num_sentences = SENTENCE_LENGTH, train_percentage = TRAIN_PERCENTAGE)
    source_seq, target_seq, target_labels, train_set, test_set, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size = data(DATASET_PATH)

    src_embed = InputEmbedding(D_MODEL, enc_vocab_size)
    input_embed = src_embed.embedding(source_seq)

    src_pos = PositionalEncoding(D_MODEL, enc_seq_length)
    final_embed = src_pos(input_embed)

    src_multi_head_attention = MultiHeadAttention(D_MODEL, HEADS)
    attention_out  = src_multi_head_attention(final_embed, final_embed, final_embed)

    src_layer_norm_1 = LayerNorm(D_MODEL)
    normalized_out_1 = src_layer_norm_1(attention_out)

    src_ff = FeedForward(D_MODEL, D_FF)
    ff_out = src_ff(normalized_out_1)

    src_layer_norm_2 = LayerNorm(D_MODEL)
    normalized_out_2 = src_layer_norm_2(ff_out)

    print(f"Encoder Output: {normalized_out_2}\n")
    print(f"Encoder Output Shape: {normalized_out_2.shape}")

def construct_decoder():
    data = DataPreparation(num_sentences = SENTENCE_LENGTH, train_percentage = TRAIN_PERCENTAGE)
    source_seq, target_seq, target_labels, train_set, test_set, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size = data(DATASET_PATH)
    
    target_embed = InputEmbedding(D_MODEL, dec_vocab_size)
    input_embed = target_embed.embedding(target_seq)

    target_pos = PositionalEncoding(D_MODEL, dec_seq_length)
    final_embed = target_pos(input_embed)

    target_masked_multi_head_attention = MultiHeadAttention(D_MODEL, HEADS, masked = True)
    attention_out_1 = target_masked_multi_head_attention(final_embed, final_embed, final_embed)

    target_layer_norm_1 = LayerNorm(D_MODEL)
    normalized_out_1 = target_layer_norm_1(attention_out_1)

    target_multi_head_attention = MultiHeadAttention(D_MODEL, HEADS, masked = True)
    attention_out_2 = target_multi_head_attention(normalized_out_1, normalized_out_1, normalized_out_1)

    target_layer_norm_2 = LayerNorm(D_MODEL)
    normalized_out_2 = target_layer_norm_2(attention_out_2)

    target_ff = FeedForward(D_MODEL, D_FF)
    ff_out = target_ff(normalized_out_2)

    target_layer_norm_3 = LayerNorm(D_MODEL)
    normalized_out_3 = target_layer_norm_3(ff_out)

    print(f"Decoder Output: {normalized_out_3}\n")
    print(f"Decoder Output Shape: {normalized_out_3.shape}")

def main():
    # construct_vocabulary()
    # construct_input_embeddings()
    # construct_positional_encodings()
    # construct_multi_head_attention_matrix()
    # perform_layer_normalization()
    construct_encoder()
    construct_decoder()

if __name__ == "__main__":
    main()