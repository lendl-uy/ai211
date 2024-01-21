# AI 211 MACHINE PROBLEM 
# Code Authors
# [1] Jan Lendl R. Uy, 2019-00312
# [2] Ryan Roi Cayas, 2022-22085
# Section: AI 211 FZZQ

# Transformer Implementation Tests

from Transformer import *
from Transformer_Constants import *
from Operational_Blocks import *
from Data_Preparation import *


def construct_vocabulary():
    data = DataPreparation(num_sentences = SENTENCE_LENGTH, train_percentage = TRAIN_PERCENTAGE)
    source_seq, target_seq, target_labels, train_set, test_set, enc_vocab_size, dec_vocab_size, enc_vocab, dec_vocab = data(DATASET_PATH)

    print(f"train set: \n{train_set}\n")

    print(f"train_enc: \n{source_seq}\n")
    print(f"enc_seq_length: {MAX_SEQ_LENGTH}, enc_vocab_size: {enc_vocab_size}\n")

    print(f"train_dec: \n{target_seq}\n")
    print(f"dec_seq_length: {MAX_SEQ_LENGTH}, dec_vocab_size: {dec_vocab_size}")

def construct_input_embeddings():
    data = DataPreparation(num_sentences = SENTENCE_LENGTH, train_percentage = TRAIN_PERCENTAGE)
    source_seq, target_seq, target_labels, train_set, test_set, enc_vocab_size, dec_vocab_size, enc_vocab, dec_vocab = data(DATASET_PATH)

    print(f"train_enc: \n{source_seq}\n")

    src_embed = InputEmbedding(D_MODEL, enc_vocab_size)
    print(f"Input Embedding: \n{src_embed.embedding(source_seq)}\n")
    print("Shape: ",src_embed.embedding(source_seq).shape)

def construct_positional_encodings():
    data = DataPreparation(num_sentences = SENTENCE_LENGTH, train_percentage = TRAIN_PERCENTAGE)
    source_seq, target_seq, target_labels, train_set, test_set, enc_vocab_size, dec_vocab_size, enc_vocab, dec_vocab = data(DATASET_PATH)

    print(f"train_enc: \n{source_seq}\n")
    print(f"Enc_seq_length: {MAX_SEQ_LENGTH}\n")

    src_embed = InputEmbedding(D_MODEL, enc_vocab_size)
    input_embed = src_embed.embedding(source_seq)
    print(f"\n Input Embedding:\n {input_embed}\n")

    src_pos = PositionalEncoding(D_MODEL, MAX_SEQ_LENGTH)
    final_embed = src_pos.forward(input_embed)
    print(f"\n Final Embedding:\n {final_embed}")
    print(f"Final Embedding Shape: {final_embed.shape}")

def construct_multi_head_attention_matrix(masked = False):
    data = DataPreparation(num_sentences = SENTENCE_LENGTH, train_percentage = TRAIN_PERCENTAGE)
    source_seq, target_seq, target_labels, train_set, test_set, enc_vocab_size, dec_vocab_size, enc_vocab, dec_vocab = data(DATASET_PATH)

    src_embed = InputEmbedding(D_MODEL, enc_vocab_size)
    input_embed = src_embed.embedding(source_seq)

    src_pos = PositionalEncoding(D_MODEL, MAX_SEQ_LENGTH)
    final_embed = src_pos.forward(input_embed)

    src_multi_attention = MultiHeadAttention(D_MODEL, HEADS, DROPOUT_PERCENT, masked)
    attention_out = src_multi_attention.forward(final_embed, final_embed, final_embed)

    print(f"attention_out: \n{attention_out}\n")
    print(f"Attention Out Shape: {attention_out.shape}")

def perform_layer_normalization():
    data = DataPreparation(num_sentences = SENTENCE_LENGTH, train_percentage = TRAIN_PERCENTAGE)
    source_seq, target_seq, target_labels, train_set, test_set, enc_vocab_size, dec_vocab_size, enc_vocab, dec_vocab = data(DATASET_PATH)

    src_embed = InputEmbedding(D_MODEL, enc_vocab_size)
    input_embed = src_embed.embedding(source_seq)

    src_pos = PositionalEncoding(D_MODEL, MAX_SEQ_LENGTH)
    final_embed = src_pos.forward(input_embed)

    src_multi_attention = MultiHeadAttention(D_MODEL, HEADS, DROPOUT_PERCENT,  mask = True)
    attention_out  = src_multi_attention.forward(final_embed, final_embed, final_embed)

    src_layer_norm = Normalization(D_MODEL)
    normalized_out = src_layer_norm.forward(attention_out)

    print(f"Normalized Out: \n{normalized_out}\n")
    print(f"Normalized Out Shape: {normalized_out.shape}")

def construct_encoder():
    data = DataPreparation(num_sentences = SENTENCE_LENGTH, train_percentage = TRAIN_PERCENTAGE)
    source_seq, target_seq, target_labels, train_set, test_set, enc_vocab_size, dec_vocab_size, enc_vocab, dec_vocab = data(DATASET_PATH)

    src_embed = InputEmbedding(D_MODEL, enc_vocab_size)
    input_embed = src_embed.embedding(source_seq)

    src_pos = PositionalEncoding(D_MODEL, MAX_SEQ_LENGTH)
    final_embed = src_pos.forward(input_embed)

    src_multi_head_attention = MultiHeadAttention(D_MODEL, HEADS, DROPOUT_PERCENT)
    attention_out  = src_multi_head_attention.forward(final_embed, final_embed, final_embed)

    src_layer_norm_1 = Normalization(D_MODEL)
    normalized_out_1 = src_layer_norm_1.forward(attention_out)

    src_ff = Feedforward(D_MODEL, D_FF)
    ff_out = src_ff.forward(normalized_out_1)

    src_layer_norm_2 = Normalization(D_MODEL)
    normalized_out_2 = src_layer_norm_2.forward(ff_out)

    print(f"Encoder Output: \n{normalized_out_2}\n")
    print(f"Encoder Output Shape: {normalized_out_2.shape}")

def construct_decoder():
    data = DataPreparation(num_sentences = SENTENCE_LENGTH, train_percentage = TRAIN_PERCENTAGE)
    source_seq, target_seq, target_labels, train_set, test_set, enc_vocab_size, dec_vocab_size, enc_vocab, dec_vocab = data(DATASET_PATH)
    
    target_embed = InputEmbedding(D_MODEL, dec_vocab_size)
    input_embed = target_embed.embedding(target_seq)

    target_pos = PositionalEncoding(D_MODEL, MAX_SEQ_LENGTH)
    final_embed = target_pos.forward(input_embed)

    target_masked_multi_head_attention = MultiHeadAttention(D_MODEL, HEADS, DROPOUT_PERCENT,  mask = True)
    attention_out_1 = target_masked_multi_head_attention.forward(final_embed, final_embed, final_embed)

    target_layer_norm_1 = Normalization(D_MODEL)
    normalized_out_1 = target_layer_norm_1.forward(attention_out_1)

    target_multi_head_attention = MultiHeadAttention(D_MODEL, HEADS, DROPOUT_PERCENT, mask = False)
    attention_out_2 = target_multi_head_attention.forward(normalized_out_1, normalized_out_1, normalized_out_1)

    target_layer_norm_2 = Normalization(D_MODEL)
    normalized_out_2 = target_layer_norm_2.forward(attention_out_2)

    target_ff = Feedforward(D_MODEL, D_FF)
    ff_out = target_ff.forward(normalized_out_2)

    target_layer_norm_3 = Normalization(D_MODEL)
    normalized_out_3 = target_layer_norm_3.forward(ff_out)

    print(f"Decoder Output: \n{normalized_out_3}\n")
    print(f"Decoder Output Shape: {normalized_out_3.shape}")

def construct_predicted_sequence():
    data = DataPreparation(num_sentences = SENTENCE_LENGTH, train_percentage = TRAIN_PERCENTAGE)
    source_seq, target_seq, target_labels, train_set, test_set, enc_vocab_size, dec_vocab_size, enc_vocab, dec_vocab = data(DATASET_PATH)
    
    target_embed = InputEmbedding(D_MODEL, dec_vocab_size)
    input_embed = target_embed.embedding(target_seq)

    target_pos = PositionalEncoding(D_MODEL, MAX_SEQ_LENGTH)
    final_embed = target_pos.forward(input_embed)

    target_masked_multi_head_attention = MultiHeadAttention(D_MODEL, HEADS, DROPOUT_PERCENT, mask = True)
    attention_out_1 = target_masked_multi_head_attention.forward(final_embed, final_embed, final_embed)

    target_layer_norm_1 = Normalization(D_MODEL)
    normalized_out_1 = target_layer_norm_1.forward(attention_out_1)

    target_multi_head_attention = MultiHeadAttention(D_MODEL, HEADS, DROPOUT_PERCENT, mask = False)
    attention_out_2 = target_multi_head_attention.forward(normalized_out_1, normalized_out_1, normalized_out_1)

    target_layer_norm_2 = Normalization(D_MODEL)
    normalized_out_2 = target_layer_norm_2.forward(attention_out_2)

    target_ff = Feedforward(D_MODEL, D_FF)
    ff_out = target_ff.forward(normalized_out_2)

    target_layer_norm_3 = Normalization(D_MODEL)
    normalized_out_3 = target_layer_norm_3.forward(ff_out)


    print(f'Target Sentences: \n{train_set[:,1]}\n')
    print(f"Target Sequence: \n{target_seq}\n")
    print(f"Max Seq Length: {MAX_SEQ_LENGTH}, Vocab Size: {dec_vocab_size}\n")

    # Final Linear Layer for Output
    final_linear_layer = Linear(D_MODEL, dec_vocab_size)
    output_logits = final_linear_layer.forward(normalized_out_3)
    print(f"Output Logits: \n{output_logits}\n")
    print(f"Output Logits Shape: {output_logits.shape}\n")

    # Apply Softmax to get probabilities
    softmax = Softmax()
    output_probs = softmax.forward(output_logits)
    print(f"Output Probabilities: \n{output_probs}\n")
    print(f"Output Probabilities Shape: {output_probs.shape}\n")

    # Get the index of the class with the highest probability
    predicted_class = np.argmax(output_probs, axis=-1)
    print(f'Predicted Indices: \n{predicted_class}\n')

    # Get the corresponding sequences of the predicted indices
    predicted_seqs = dec_vocab.idx_to_seq(predicted_class)
    print(f'Predicted Sequences:')
    for seq in predicted_seqs:
        print(seq)

    




def main():
    # comment out functions one at a time


    construct_vocabulary()
    # construct_input_embeddings()
    # construct_positional_encodings()
    # construct_multi_head_attention_matrix()
    # construct_multi_head_attention_matrix(masked=True)
    # perform_layer_normalization()
    # construct_encoder()
    # construct_decoder()
    # construct_predicted_sequence()

    # For backprop: run transformer_train and verify that it runs successfully and the loss is decreasing per epoch



if __name__ == "__main__":
    main()