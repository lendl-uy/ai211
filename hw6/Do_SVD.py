# Code Author: Jan Lendl R. Uy
# Student Number: 2019-00312
# Class: AI 211 FZZQ

# Coding Challenge 6: Perform Bidiagonalization and Tridiagonalization of a Matrix

# Useful references:
# [1] http://www.math.iit.edu/~fass/477577_Chapter_12.pdf

import numpy as np
import scipy as sp

EPSILON = 1e-12

np.set_printoptions(precision=2)
    
def get_bidiagonal_matrix(A):
    
    rows, cols = A.shape
    
    B = A.copy()
    
    # Golub-Kahan Bidiagonalization
    # Essentially obtaining Householder transformation for each column AND
    # row of the input matrix
    for i in range(cols-1):
        
        x = B[i:, i]
        e = np.zeros(x.size)
        
        if abs(x[0]) < EPSILON:
            e[0] = 1.0
            u = x + np.linalg.norm(x)*e
        else:
            e[0] = np.sign(x[0])
            u = x + np.sign(x[0])*np.linalg.norm(x)*e
            
        u *= 1/np.linalg.norm(u)
        u = u[:][np.newaxis].T # Cast 1D array to a column vector of a 2D array
        
        # Column-wise Householder transformation
        B[i:, i:] = B[i:, i:] - 2*u @ (u.T @ B[i:, i:])
        
        if i < cols-2:
            x = B[i, i+1:]
            e = np.zeros(x.size)
        
            if abs(x[0]) < EPSILON:
                e[0] = 1.0
                v = x + np.linalg.norm(x)*e
            else:
                e[0] = np.sign(x[0])
                v = x + np.sign(x[0])*np.linalg.norm(x)*e
                
            v *= 1/np.linalg.norm(v)
            v = v[:][np.newaxis].T # Cast 1D array to a row vector of a 2D array
            
            # Row-wise Householder transformation
            B[i:, i+1:] = B[i:, i+1:] - 2*(B[i:, i+1:] @ v) @ v.T

    return B
    
def get_tridiagonal_matrix(A):
    
    rows, cols = A.shape
    
def check_if_bidiagonal(B):
    
    rows, cols = B.shape
    
    # Verify if matrix is bidiagonal by reconstructing B but only with the
    # diagonal and bidiagonal entries. Difference between B and reconstructed B 
    # must NOT result in the zero matrix
    diagonal = np.diag(B)
    superdiagonal = np.diag(B, k=1)
    B_reconstructed = np.diag(diagonal, k=0) + np.diag(superdiagonal, k=1)
    
    difference = np.sum(B-B_reconstructed)
    if not np.allclose(difference, np.zeros((rows, cols), dtype="float"), 
                    rtol=1e-3, atol=1e-5):
        raise RuntimeError(f"Matrix is not bidiagonal!")

def compare_singular_values(A, B):
    
    U_A, S_A, Vt_A = np.linalg.svd(A)
    U_B, S_B, Vt_B = np.linalg.svd(B)
    
    if S_A.size != S_B.size:
        raise RuntimeError(f"Number of singular values do not match!")        
    
    if not np.allclose(S_A-S_B, np.zeros(S_A.size, dtype="float"), 
                    rtol=1e-3, atol=1e-5):
        raise RuntimeError(f"Singular values of B are not equal to the singular values of A")

def verify_bidiagonal_matrix(A, B, suppress_success_flag=False):
    
    # Must check if B is really bidiagonal and if its singular values are the
    # same as that of A 
    check_if_bidiagonal(B)
    compare_singular_values(A, B)
    print(f"Bidiagonal matrix is correct!")