# AI 211 MACHINE PROBLEM 
# Code Authors
# [1] Jan Lendl R. Uy, 2019-00312
# [2] Ryan Roi Cayas, 2023-XXXXX
# Section: AI 211 FZZQ

# Transformer Implementation Tests

from Transformer import *
from Transformer_Train import *
    
def test_Vocab():
    data = DataPrep(num_sentences = 5, train_percentage = 0.7)
    train_enc, train_dec, train_set, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size = data('english-german-both.pkl')
    print("train set:")
    print(train_set)
    print()

    print("train_enc:")
    print(train_enc)
    print(f"enc_seq_length: {enc_seq_length}, enc_vocab_size: {enc_vocab_size}")
    print()

    print("train_dec:")
    print(train_dec)
    print(f"dec_seq_length: {dec_seq_length}, dec_vocab_size: {dec_vocab_size}")

def test_InputEmbed():
    data = DataPrep(num_sentences = 5, train_percentage = 0.7)
    train_enc, train_dec, train_set, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size = data('english-german-both.pkl')

    print("train_enc:")
    print(train_enc)
    print()

    d_model = 10
    src_embed = InputEmbedding(d_model,enc_vocab_size)
    print("Input Embedding:\n")
    print(src_embed.embedding(train_enc))
    print("Shape: ",src_embed.embedding(train_enc).shape)

def test_PosEmbed():
    data = DataPrep(num_sentences = 5, train_percentage = 0.7)
    train_enc, train_dec, train_set, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size = data('english-german-both.pkl')

    print("train_enc:")
    print(train_enc)
    print()

    print("Enc_seq_length: ",enc_seq_length)

    d_model = 10
    src_embed = InputEmbedding(d_model,enc_vocab_size)
    input_embed = src_embed.embedding(train_enc)
    print("\nInput Embedding:\n")
    print(input_embed)

    src_pos = PositionalEncoding(d_model,enc_seq_length)
    final_embed = src_pos(input_embed)
    print("\nFinal Embedding:\n",final_embed)

def main():
    test_PosEmbed()

if __name__ == "__main__":
    main()