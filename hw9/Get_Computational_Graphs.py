# Code Author: Jan Lendl R. Uy
# Student Number: 2019-00312
# Class: AI 211 FZZQ

# Coding Challenge 9: Get Computational Graph

# Useful references:

import numpy as np

np.set_printoptions(precision=2)

def get_computational_graph(A):
    
    rows, cols = A.shape
    
def verify_computational_graph(A, suppress_success_flag=False):
    
    rows, cols = A.shape
        
    if not np.allclose(A, np.zeros((rows, cols), dtype="float"), atol=1e-2):
        raise RuntimeError(f"One or more factors of the decomposition are incorrect!")
    
    if not suppress_success_flag:
        print(f"Computed factors from SVD are correct!")