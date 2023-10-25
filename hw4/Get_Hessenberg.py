# Code Author: Jan Lendl R. Uy
# Student Number: 2019-00312
# Class: AI 211 FZZQ

# Coding Challenge 4: Compute Hessenberg Matrix of a Square Matrix

# Useful reference for algorithm: https://www.youtube.com/watch?v=t_bj3V9Ubac

import numpy as np

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


def verify_hessenberg_matrix(A, Hess, H, suppress_success_flag=False):

    rows, cols = A.shape
    
    # Verification of computed Hessenberg matrix
    H_Hess_H = H @ Hess @ np.linalg.inv(H)
        
    # allclose() is used due to tendency of numpy to introduce negative sign to zeros
    # and -0.0 != 0.0 by convention
    if np.allclose(np.subtract(A, H_Hess_H), np.zeros((rows, cols), dtype="float"), 
                   rtol=1e-3, atol=1e-5):
        if not suppress_success_flag:
            print("Computed Hessenberg matrix is correct!")
    else:
        print(f"A = {A}\n")
        print(f"H @ Hess @ H^-1 = {H_Hess_H}")
        print(f"Hess = {Hess}\n")
        print(f"H = {H}\n")
        raise RuntimeError("Computed Hessenberg matrix is incorrect!")