# Code Author: Jan Lendl R. Uy
# Student Number: 2019-00312
# Class: AI 211 FZZQ

# Coding Challenge 5: Compute Eigenvalues and Eigenvectors of a Matrix via QR Algorithm

# Useful references:
# https://faculty.ucmerced.edu/mhyang/course/eecs275/lectures/lecture17.pdf
# http://web.stanford.edu/class/cme335/lecture5
# For Wilkinson shift: https://math.stackexchange.com/questions/1262363/convergence-of-qr-algorithm-to-upper-triangular-matrix

import numpy as np
import math
import scipy as sp

MAX_QR_ITERATIONS = 2000
QR_TOLERANCE = 1e-10
EPSILON = 1e-7

np.set_printoptions(precision=3)

def is_symmetric(A):
    
    return np.array_equal(A, A.T)

def is_hessenberg(A):
    
    rows, cols = A.shape # Get the number of rows and columns of A
    
    # Search lower triangle below the subdiagonal if all are zeros
    for col in range(cols-2):
        for row in range(col+2, rows):
            mtx_elem = A[row, col]
            if abs(mtx_elem) >= EPSILON:
                return False
    return True  

def get_hessenberg_matrix(A):
        
    rows, cols = A.shape # Get the number of rows and columns of A

    # Catch non-square matrix inputs
    if rows != cols:
        raise ValueError("Input matrix is not a square!")
    elif (rows == cols == 2) or is_hessenberg(A):
        return A, np.identity(rows)

    Hess = A.copy() # Get a copy of A so as not to mutate contents of A
    H = np.identity(rows) # Initialize Householder matrix
    
    # Obtain Householder reflections for columns-2 iterations
    # Final 2x2 submatrix is already in its upper Hessenberg matrix
    for i in range(cols-2):
        mtx_col = Hess[:, i]
        x = mtx_col[i+1:] # Get the sub-row vector x        
        u = np.zeros(x.size)
        
        # If x[0] = 0, set sgn(x[0]) = 1
        if x[0] == 0.0:
            u[0] = -np.linalg.norm(x) # First entry is set to -sgn(x_0)l2_norm(x)
        else:
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

def wilkinson_shift(A, block_pos=0):
    
    rows, cols = A.shape # Get the number of rows and columns of A
    
    # Extract the 2x2 block matrix B at the bottom left of the matrix
    # Compute "estimated eigenvalue" (mu) from B
    B = A[rows-2-block_pos: rows-block_pos, cols-2-block_pos: cols-block_pos]
    #print(f"B = {B}")
    a = B[0, 0]
    b_10 = B[1, 0]
    b_01 = B[0, 1]
    c = B[1, 1]
    l = (a-c)/2
    
    # If l = 0, set sgn(l) = 1
    if l == 0.0:
        µ = c - (b_10*b_01)/(abs(l) + math.hypot(l, b_10*b_01)) # Shift coefficient
    else:
        µ = c - (np.sign(l)*b_10*b_01)/(abs(l) + math.hypot(l, b_10*b_01)) # Shift coefficient

    return µ
    
def get_eigenvalues(A, tol=QR_TOLERANCE):
        
    # Convert A to Hessenberg matrix to reduce computational complexity    
    Hess, H = get_hessenberg_matrix(A)
        
    rows, cols = Hess.shape # Get the number of rows and columns of Hess

    # Set initial value of eigenvectors to H
    # This is so that final eigenmatrix is that of A, not Hess
    eigenvectors = H
    
    iterations = 0
    
    # Perform QR algorithm to compute for eigenvalues until 
    # Schur form is (approximately) achieved
    #print(f"Performing QR algorithm ...")
    for n in range(cols-1):
        
        subdiagonal_elem = Hess[rows-1-n, cols-2-n]
        
        # Start zeroing out subdiagonal entries from the bottom-right of matrix
        # for faster convergence
        while abs(subdiagonal_elem) > tol:
                                
            # Compute Wilkinson shift
            µ = wilkinson_shift(Hess, n)

            # Perform similarity transformation to obtain a Schur matrix
            Q, R = qr_via_givens(Hess - µ*np.identity(rows))
            Hess = R @ Q + µ*np.identity(rows)
            
            Q_H = np.identity(rows)
            Q_H[:rows-n, :cols-n] = Q[:rows-n, :cols-n]
            eigenvectors = eigenvectors @ Q_H
            
            # Terminate algorithm if Hess is not converging to a Schur matrix
            if iterations >= MAX_QR_ITERATIONS:
                raise RuntimeError(f"QR Algorithm was not able to converge for a tolerance of {tol} and max iterations of {MAX_QR_ITERATIONS}")
            
            # Update subdiagonal element and iteration count
            subdiagonal_elem = Hess[rows-1-n, cols-2-n]
            iterations += 1
            
        # Deflate the matrix after finding an eigenvalue
        Hess[rows-n-1, :cols-n-1] = np.zeros(cols-n-1) # Set (row-n)th row to zeros
        Hess[:rows-n-1, cols-n-1] = np.zeros(rows-n-1) # Set (col-n)th column to zeros
        
    # Diagonal entries of the Hessenberg matrix are the eigenvalues of A
    eigenvalues = np.diag(Hess)
    
    #print(f"iterations = {iterations}")
        
    if is_symmetric(A):
        #print(f"A is symmetric! Eigenvectors will also be returned.")
        return eigenvalues, eigenvectors

    return eigenvalues

def verify_eigenvalues(A, eigenvalues, suppress_success_flag=False):
    
    rows, cols = A.shape
    
    '''
    # Cross-check computed eigenvalues to that of numpy   
    eigenvalues_from_np = np.linalg.eigvals(A)
    '''
    
    # Verify correctness of eigenvalues such that trace(A) = Σλ
    for n in range(cols):
        trace_A = np.trace(A)
        sum_eigenvalues = np.sum(eigenvalues)
        if abs(trace_A-sum_eigenvalues) > 1e-4:
            raise RuntimeError(f"One of the computed eigenvalues is incorrect!")
        
    if not suppress_success_flag:
        print("All computed eigenvalues are correct!")

def verify_eigenvectors(A, eigenvalues, eigenvectors, suppress_success_flag=False):
    
    rows, cols = A.shape
    
    '''
    # Cross-check computed eigenvalues to that of numpy   
    eigenvalues_np, eigenvectors_np = np.linalg.eig(A)
    print(f"eigenvectors_np = {eigenvectors_np}")
    print(f"eigenvectors = {eigenvectors}")
    '''
    
    # Verify correctness of eigenvectors by solving (A-λI)v = 0
    # allclose() is used for comparison due to floating point precision errors 
    # and tendency of numpy to introduce negative sign to zeros and 
    # -0.0 != 0.0 by convention
    for n in range(cols):
        
        eigenvector_equation = (A - eigenvalues[n]*np.identity(rows)) @ eigenvectors[:, n]
        if not np.allclose(eigenvector_equation, np.zeros(rows, dtype="float"), 
                    rtol=1e-1, atol=1e-1):
            raise RuntimeError(f"The computed eigenvector {eigenvectors[:, n]} is incorrect!")
        
    if not suppress_success_flag:
        print("All computed eigenvectors are correct!")