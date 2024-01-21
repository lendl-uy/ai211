# Transformer Training
# Sequence to Sequence Model - English to German
# WMT English-German 2014 Dataset

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


from Transformer_Constants import *
from Transformer import Transformer
from Data_Preparation import DataPreparation
from Operational_Blocks import CrossEntropy

data = DataPreparation(num_sentences = SENTENCE_LENGTH, train_percentage = TRAIN_PERCENTAGE)
source_seq, target_seq, target_labels, train_set, test_set, enc_vocab_size, dec_vocab_size, enc_vocab, dec_vocab = data('english-german-both.pkl')

# print(f"source_seq = {source_seq}")
# print(f"target_seq = {target_seq}")
# print(f"target_labels = {target_labels}")
# print(f"train_set = {train_set}")
# print(f"test_set = {test_set}")
print(f"enc_vocab_size = {enc_vocab_size}")
print(f"dec_vocab_size = {dec_vocab_size}")

model = Transformer(d_model = D_MODEL, num_heads = HEADS, d_ff = D_FF, 
                    source_seq_length = MAX_SEQ_LENGTH, target_seq_length = MAX_SEQ_LENGTH, 
                    source_vocab_size = enc_vocab_size, target_vocab_size = dec_vocab_size, 
                    learning_rate = LEARNING_RATE)

cross_entropy = CrossEntropy()

train_losses = []
test_losses = []

# For testing
test_source_seq = enc_vocab.seq_to_idx(test_set[:,0], MAX_SEQ_LENGTH)
test_target_seq = dec_vocab.seq_to_idx(test_set[:,1], MAX_SEQ_LENGTH)

for epoch in range(EPOCHS):

    total_loss = []
    batch = 0

    # Initialize printing of loss per batch
    tqdm_range = tqdm( enumerate(zip(source_seq, target_seq)), total = len(source_seq))
    
    # Iterate over batches
    for i in range(0, len(source_seq), BATCH_SIZE):
        batch_source_seq = source_seq[i:i + BATCH_SIZE]
        batch_target_seq = target_seq[i:i + BATCH_SIZE]
    
        # print(f"batch_source_seq = {batch_source_seq}")
        # print(f"batch_target_seq = {batch_target_seq}")

        # Forward pass
        model_output = model.forward(batch_source_seq, batch_target_seq)
        # print(f"model_output = {model_output}")

        # target_labels = np.eye(dec_vocab_size)[batch_target_seq]
        # print(f"batch_target_seq = {batch_target_seq}")

        # Reshape model output to 2D and target labels to 1D 
        # for computation of loss
        i, j, k = model_output.shape
        model_output_reshaped = np.reshape(model_output, (i*j, k))
        batch_target_seq_flat = np.array(batch_target_seq).astype(np.int32).flatten()

        # Compute the loss
        loss = cross_entropy.loss(model_output_reshaped, batch_target_seq_flat).mean()
        total_loss.append(loss)

        # Backward pass
        grad_upstream = cross_entropy.derivative(model_output_reshaped, batch_target_seq_flat)
        grad_upstream_reshaped = grad_upstream.reshape(model_output.shape)
        model.backward(grad_upstream_reshaped, batch_source_seq, batch_target_seq)

        # Update progress bar
        tqdm_range.set_description(f"TRAIN | Epoch {epoch + 1}/{EPOCHS} | Cross Entropy Loss: {loss:.3f}")
        tqdm_range.update(1)
        
        # print(f"Done Batch {batch+1}, Loss: {loss}\n")
        batch += 1

    average_loss = np.mean(total_loss)
    train_losses.append(average_loss)

    # compute test loss
    test_model_output = model.forward(test_source_seq, test_target_seq)

    # Reshape model output to 2D and target labels to 1D 
    # for computation of loss
    i, j, k = test_model_output.shape
    test_model_output_reshaped = np.reshape(test_model_output, (i*j, k))
    test_target_seq_flat = np.array(test_target_seq).astype(np.int32).flatten()

    # Compute the loss
    test_loss = cross_entropy.loss(test_model_output_reshaped, test_target_seq_flat).mean()
    test_losses.append(test_loss)

    print(f'Epoch {epoch + 1}, Average Train Loss: {average_loss}, Test Loss: {test_loss}')

# Plotting the losses
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Testing Loss Over Epochs')
plt.legend()
plt.show()
