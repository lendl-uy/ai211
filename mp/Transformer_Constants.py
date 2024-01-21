# Transformer Architecture Constants

# Path to the dataset
DATASET_PATH = "english-german-both.pkl"

# Training constants
EPOCHS = 5
BATCH_SIZE = 2 # 32 original value
TRAIN_PERCENTAGE = 0.8
DROPOUT_PERCENT = 0.1

# Input constants
SENTENCE_LENGTH = 5 # 10,000 original value - number of sentences to include in data
MAX_SEQ_LENGTH = 10

# Model hyperparameters
LEARNING_RATE = 0.01
D_MODEL = 8 # 512 original value (should be divisible by # of heads and ideally even)
D_FF = 12 # 2048 original value
HEADS = 4