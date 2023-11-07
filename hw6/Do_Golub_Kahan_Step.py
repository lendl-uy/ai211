# Code Author: Jan Lendl R. Uy
# Student Number: 2019-00312
# Class: AI 211 FZZQ

# Coding Challenge 6: Perform Bidiagonalization and Tridiagonalization of a Matrix

# Useful references:
# [1] http://www.math.iit.edu/~fass/477577_Chapter_12.pdf
# [2] https://mathoverflow.net/questions/347723/using-permutation-matrix-to-convert-a-matrix-into-tridiagonal-matrix

import numpy as np
import math

EPSILON = 1e-12

np.set_printoptions(precision=1, suppress=True)

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

def zero_singular_values_exist(S):
    
    for i in range(S.size):
        if abs(S[i]) < EPSILON:
            return True
    return False

def compare_singular_values(A, B, remove_duplicates=False):
    
    U_A, S_A, Vt_A = np.linalg.svd(A)
    U_B, S_B, Vt_B = np.linalg.svd(B)
    
    if remove_duplicates:
        # Round off to nearest 5 decimal places due to precision errors
        # Remove erroneous zero formed by augmented matrix H and
        # duplicates, if any, then sort in descending order
        np.around(S_B, 5, S_B)
        if not zero_singular_values_exist(S_A):   
            index = np.argwhere(abs(S_B) < EPSILON)
            S_B = np.delete(S_B, index)
        S_B = np.flip(np.unique(S_B))
            
    if S_A.size != S_B.size:
        #print(f"A = {A}")
        #print(f"B = {B}")
        #print(f"S_A = {S_A}")
        #print(f"S_B = {S_B}")
        raise RuntimeError(f"Number of singular values do not match!")        
    
    if not np.allclose(S_A-S_B, np.zeros(S_A.size, dtype="float"), 
                    rtol=1e-3, atol=1e-5):
        #print(f"A = {A}")
        #print(f"B = {B}")
        #print(f"S_A = {S_A}")
        #print(f"S_B = {S_B}")
        raise RuntimeError(f"Singular values of computed matrix are not equal to the singular values of A")

def verify_bidiagonal_matrix(A, B, suppress_success_flag=False):
    
    # Must check if B has the singular values as that of A 
    compare_singular_values(A, B)
    if not suppress_success_flag:
        print(f"Computed bidiagonal matrix is correct!")
    
def verify_tridiagonal_matrix(A, T, suppress_success_flag=False):
    
    # Must check if T has the singular values as that of A 
    # Remove duplicate singular values since duplicates are introduced by the
    # augmented matrix H in the tridiagonalization process
    compare_singular_values(A, T, remove_duplicates=True)
    if not suppress_success_flag:
        print(f"Computed tridiagonal matrix is correct!")