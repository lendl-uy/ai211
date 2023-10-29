# Code Author: Jan Lendl R. Uy
# Student Number: 2019-00312
# Class: AI 211 FZZQ

# Coding Challenge 5: Compute Eigenvalues and Eigenvectors of a Matrix via QR Algorithm

import numpy as np
import math

def get_hessenberg_matrix(A):
        
    rows, cols = A.shape # Get the number of rows "m" and columns "n"
    
    # Catch non-square matrix inputs
    if rows != cols:
        raise ValueError("Input matrix is not a square!")

    Hess = A.copy() # Get a copy of A so as not to mutate contents of A
    H = np.identity(rows) # Initialize Householder matrix
    
    # Obtain Householder reflections for columns-2 iterations
    # Final 2x2 submatrix is already in its upper Hessenberg matrix
    for i in range(cols-2):
        mtx_col = Hess[:, i]
        x = mtx_col[i+1:] # Get the sub-row vector x        
        u = np.zeros(x.size)
        u[0] = -np.sign(x[0])*np.linalg.norm(x) # First entry is set to -sgn(x_0)l2_norm(x)
        v = u-x
        v = v[:][np.newaxis].T # Convert v to a 2D numpy array and get its tranpose to convert to column vector
        v_t = v.T # Tranpose of v
        
        # Handle case when column vector is a zero vector, avoid dividing by zero
        if np.sum(np.absolute(x)) == 0.0:
            p = v @ v_t
        else:
            p = v @ v_t/(v_t @ v)
        
        # Get the Householder matrix
        H_temp = np.identity(rows) # Initialize the Householder matrix for the current iteration
        h = np.identity(x.size) - 2*p # Get the sub-Househoulder matrix
        H_temp[i+1:, i+1:] = h
        Hess = H_temp @ Hess @ H_temp # HAH^-1 is equal to HAH
        H = H @ H_temp # Update the Householder matrix, i.e. H = H_n @ H_n-1 ... @ H_1
        
    Hess = np.around(Hess, 7) # Round of entries of Hess to filter out floating point errors

    return Hess, H

def givens_rotation(A, i, j):
    
    rows, cols = A.shape # Get the number of rows "m" and columns "n" of A
    
    G = np.eye(rows, cols) # Initialize the Givens rotation matrix
    
    # Set the 2x2 elements at which rotation will be performed
    a = A[j, j]
    b = A[i, j]
    r = math.hypot(a, b)
    
    G[i, i], G[j, j] = a/r, a/r
    G[i, j] = -b/r
    G[j, i] = b/r

    return G

def wilkinson_shift(A):
    
    rows, cols = A.shape # Get the number of rows "m" and columns "n" of A

def get_eigenvalues(A):
        
    rows, cols = A.shape # Get the number of rows "m" and columns "n" of A
    
    Hess, H = get_hessenberg_matrix(A)
    
    # Perform Givens rotation on the Hessenberg matrix to set subdiagonal elements to 0
    for i in range(cols-1):
        G = givens_rotation(Hess, i+1, i)
        Hess = G @ Hess
    
    np.around(Hess, 8) # Remove numerical errors from rational numbers     
    print(f"Hess = {Hess}")
    #print(f"H = {H}")
    
def get_eigenvectors(A):
        
    rows, cols = A.shape # Get the number of rows "m" and columns "n" of A

def verify_eigenvalues(A, suppress_success_flag=False):

    rows, cols = A.shape # Get the number of rows "m" and columns "n" of A

def verify_eigenvectors(A, suppress_success_flag=False):

    rows, cols = A.shape # Get the number of rows "m" and columns "n" of A