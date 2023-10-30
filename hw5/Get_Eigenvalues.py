# Code Author: Jan Lendl R. Uy
# Student Number: 2019-00312
# Class: AI 211 FZZQ

# Coding Challenge 5: Compute Eigenvalues and Eigenvectors of a Matrix via QR Algorithm

import numpy as np
import math

MAX_QR_ITERATIONS = 5000
QR_TOLERANCE = 1e-7

def get_hessenberg_matrix(A):
        
    rows, cols = A.shape # Get the number of rows and columns of A
    
    # Catch non-square matrix inputs
    if rows != cols:
        raise ValueError("Input matrix is not a square!")
    elif rows == cols == 2:
        return A, np.identity(rows)

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
    
    rows, cols = A.shape # Get the number of rows and columns of A
    
    G = np.eye(rows, cols) # Initialize the Givens rotation matrix
    
    # Set the 2x2 elements at which rotation will be performed
    a = A[j, j]
    b = A[i, j]
    r = math.hypot(a, b)
    
    G[i, i], G[j, j] = a/r, a/r
    G[i, j] = -b/r
    G[j, i] = b/r

    return G
    
def qr_via_givens(A):
    
    rows, cols = A.shape # Get the number of rows and columns of A
    
    Q = np.eye(rows, rows)
    R = A.copy()

    # Perform Givens rotation on the Hessenberg matrix to get Q and R matrices
    for i in range(cols-1):
        subdiagonal_elem = R[i+1, i]
        if subdiagonal_elem != 0.0:
            G = givens_rotation(R, i+1, i)
            R = G @ R
            Q = Q @ G.T
        
    return Q, R

def wilkinson_shift(A):
    
    rows, cols = A.shape # Get the number of rows and columns of A
    
    # Extract the 2x2 block matrix at the bottom left of the matrix
    # Compute "estimated eigenvalue" (mu) from this block
    block = A[rows-2: , cols-2:]
    a = block[0, 0]
    b = block[0, 1]
    c = block[1, 1]
    l = (a-c)/2
    mu = c - (np.sign(l)*b**2)/(abs(l) + math.hypot(l, b)) # Shift coefficient
    
    return mu
    
def get_eigenvalues(A, tol=QR_TOLERANCE):
        
    # Convert A to Hessenberg matrix to reduce computational complexity    
    Hess, H = get_hessenberg_matrix(A)
    
    rows, cols = Hess.shape # Get the number of rows and columns of Hess
        
    iterations = 0
        
    # Perform QR algorithm to compute for eigenvalues until 
    # Schur form is (approximately) achieved
    while np.linalg.norm(np.diag(Hess, k=-1)) > tol:
        
        # Get Wilkinson shift coefficient to speed up convergence
        mu = wilkinson_shift(Hess)
        
        Q, R = qr_via_givens(Hess - mu*np.identity(rows))
        #Q, R = qr_via_givens(Hess)
        Hess = R @ Q + mu*np.identity(rows)
        #Hess = R @ Q
        iterations += 1
        #print(f"norm of subdiag = {np.linalg.norm(np.diag(Hess, k=-1))}")
        if iterations >= MAX_QR_ITERATIONS:
            raise RuntimeError(f"QR Algorithm was not able to converge for a tolerance of {tol}")
    
    # Diagonal entries of the Hessenberg matrix are the eigenvalues of A
    eigenvalues = np.diag(Hess)
    # Sort eigenvalues in decreasing order
    eigenvalues = sorted(eigenvalues.tolist(), reverse=True)
    
    #print(f"lambda_i = {np.diag(Hess)}")
    #print(f"iterations = {iterations}")
    
    return eigenvalues
        
def get_eigenvectors(A):
        
    rows, cols = A.shape # Get the number of rows and columns of A

def verify_eigenvalues(A, eigenvalues, suppress_success_flag=False):
    
    eigenvalues_from_np = np.linalg.eigvals(A)
    eigenvalues_from_np = sorted(eigenvalues_from_np.tolist(), reverse=True)
    
    # Check if number of eigenvalues match that of numpy
    if len(eigenvalues_from_np) != len(eigenvalues):
        raise RuntimeError("The number of eigenvalues is incorrect!")
    
    # allclose() is used due to tendency of numpy to introduce negative sign 
    # to zeros and -0.0 != 0.0 by convention
    if np.allclose(np.subtract(eigenvalues_from_np, eigenvalues), 
                   np.zeros(len(eigenvalues_from_np), dtype="float"), 
                   rtol=1e-3, atol=1e-5):
        if not suppress_success_flag:
            print("Computed eigenvalues are correct!")
    else:
        raise RuntimeError("Computed eigenvalues are incorrect!")

def verify_eigenvectors(A, eigenvectors, suppress_success_flag=False):

    rows, cols = A.shape # Get the number of rows and columns of A
