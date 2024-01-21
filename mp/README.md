# AI 211 Machine Problem
## Transformer Architecture

This machine problem submission was created by the following people:
- Jan Lendl Uy, 2019-00312
- Ryan Roi Cayas, 2022-22085

Found in this project directory are the following files:

- Transformer.py
- Transformer_Constants.py
- Decoder.py
- Encoder.py
- Data_Preparation.py
- Transformer_Train.py
- Trainsformer_Inference.py
- PartialResults.py

In `Transformer.py`, it contains the transformer architecture written using the Numpy library in Python. It brings together the encoder and decoder blocks in `Encoder.py` and `Decoder.py`. Moreover, `Operational_Blocks.py` contains all the classes and functions that were needed to construct all layers in the architecture. `Data_Preparation.py` prepares the data to be used for modeling the transformer (e.g. appending of the start and end tokens and building the vocabulary). You may check these files to inspect the implementation of the transformer.

Values of the hyperparameters and other variables used in the transformer modeling process may be changed in `Transformer_Constants.py`.

`Transformer_Inference.py` contains the code that makes use of the trained transformer model to translate English sentences to German. Input the English sentence on the file and run it to see the predicted German translation.

Lastly, `PartialResults.py` contains functions which prints the outputs of intermediary layers in the transformer architecture. Run this file to check the results for different intermediary layers. All functions are initially commented out in the `main()` function of the code. Uncomment one function at a time before running the program.

