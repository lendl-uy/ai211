# Code Author: Jan Lendl R. Uy
# Student Number: 2019-00312
# Class: AI 211 FZZQ

# Coding Challenge 3: QR Decomposition with Column Pivoting

import numpy as np

EPSILON = 1e-5

def verify_hessenberg(A, Q, R, P, suppress_success_flag=False):
    
    # Verification of computed matrices
    AP = A @ P
    QR = Q @ R
    
    m, n = A.shape
    
    # allclose() is used due to loss of precision when performing row-wise operations
    if np.allclose(np.subtract(AP, QR), np.zeros((m, n), dtype="float"), rtol=1e-3, atol=1e-5):
        if not suppress_success_flag:
            print("Decomposition is correct!")
    else:
        raise RuntimeError("The decomposition is incorrect!")