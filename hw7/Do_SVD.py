# Code Author: Jan Lendl R. Uy
# Student Number: 2019-00312
# Class: AI 211 FZZQ

# Coding Challenge 7: Perform SVD

# Useful references:
# [1] http://www.math.iit.edu/~fass/477577_Chapter_12.pdf
# [2] https://mathoverflow.net/questions/347723/using-permutation-matrix-to-convert-a-matrix-into-tridiagonal-matrix

import numpy as np
import math

EPSILON = 1e-12

#np.set_printoptions(precision=5)

def zero_out_small_numbers(A, tol=EPSILON):
    
    rows, cols = A.shape
    
    for i in range(rows):
        for j in range(cols):
            mtx_elem = A[i, j]
            if abs(mtx_elem) < EPSILON:
                mtx_elem = A[i, j] = 0.0
       
def get_bidiagonal_matrix(A):
    
    rows, cols = A.shape
    
    B = A.copy()
    
    # Golub-Kahan Bidiagonalization
    # Essentially obtaining Householder transformation for each column AND
    # row of the input matrix
    if cols > rows:
        m = rows
    else:
        m = cols
        
    for i in range(m):
        
        if i < rows-1:
            x = B[i:, i]
            e = np.zeros(x.size)

            if abs(x[0]) < EPSILON:
                e[0] = 1.0
                u = x + np.linalg.norm(x)*e
            else:
                e[0] = np.sign(x[0])
                u = x + np.sign(x[0])*np.linalg.norm(x)*e
            
            # Retain u if it is the ~zero vector
            if np.sum(np.absolute(u)) > EPSILON:
                u *= 1/np.linalg.norm(u)
            u = u[:][np.newaxis].T # Cast 1D array to a column vector of a 2D array
            
            # Column-wise Householder transformation
            B[i:, i:] = B[i:, i:] - 2*u @ (u.T @ B[i:, i:])
        
        if i < cols-1:
            x = B[i, i+1:]
            e = np.zeros(x.size)

            if abs(x[0]) < EPSILON:
                e[0] = 1.0
                v = x + np.linalg.norm(x)*e
            else:
                e[0] = np.sign(x[0])
                v = x + np.sign(x[0])*np.linalg.norm(x)*e
                    
            # Retain v as is if it is the ~zero vector
            if np.sum(np.absolute(v)) > EPSILON:
                v *= 1/np.linalg.norm(v)
            v = v[:][np.newaxis].T # Cast 1D array to a row vector of a 2D array
                
            # Row-wise Householder transformation
            B[i:, i+1:] = B[i:, i+1:] - 2*(B[i:, i+1:] @ v) @ v.T
            
    zero_out_small_numbers(B)

    return B
    
def get_tridiagonal_matrix(A):
    
    B = get_bidiagonal_matrix(A)
    
    rows, cols = B.shape
    
    # Generate a larger matrix H such that H = {[0 B.T], [B 0]}
    # Permuting H properly will result in a tridiagonal matrix
    H = np.zeros((rows+cols, rows+cols))
    
    H[:cols, cols:] = B.T # Set upper right submatrix to B.T
    H[cols:, :cols] = B # Set bottom left submatrix to B
    
    # H has zero rows and/or columns when rows > cols
    # Initialize iteration var m to 2*cols < rows+cols to avoid permuting
    # these zero rows and/or columns
    if rows > cols:
        m = 2*cols
        H = H[:m, :m]
    else:
        m = rows+cols
        
    # Initialize permutation matrix P that shifts columns to achieve tridiagonal form
    # whose permutations are determined by the following piecewise equation:
    # pi = i/2, i mod 2 == 0
    # pi = floor(i/2) + ceil((rows+cols)/2), i mod 2 == 1
    P = np.zeros((m, m))
    
    for i in range(0, m):
        if i%2 == 0:
            pi = i//2 # Permutation index for even columns, // used for int division
            P[pi, i] = 1.0
        else:
            pi = i//2 + math.ceil((m)/2) # Permutation index for odd columns
            P[pi, i] = 1.0
        
    T = P.T @ H @ P
    
    return T