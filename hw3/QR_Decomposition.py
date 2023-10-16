# Code Author: Jan Lendl R. Uy
# Student Number: 2019-00312
# Class: AI 211 FZZQ

# Coding Challenge 3: QR Decomposition with Column Pivoting

import numpy as np

EPSILON = 1e-5

def rref(A):

    # Get the mxn shape of the matrix
    m, n = A.shape

    pivot = 0

    # RREF Algorithm
    # Look for the leftmost nonzero entry of each row
    # Perform row-wise operations to obtain a leading 1 and all entries in that
    # column to be zero
    for i in range(n):
        pivot_found = False # Reset pivot_found for every iteration of columns

        for j in range(m):
            mtx_elem = A[j, i]

            # Check for leftmost nonzero entry
            if mtx_elem == 0:
                continue
            else:

                # Perform row-wise operations to convert matrix into its RREF
                if j == pivot:
                    A[j, :] *= 1/mtx_elem
                    pivot_found = True

                    for k in range(m):
                        if k != pivot:
                            A[k, :] = A[k, :] - A[pivot, :]*A[k, i]
                    break
                else:

                    # Swap rows with the topmost row without a pivot yet
                    if not pivot_found and j>pivot:
                        A[j, :] *= 1/mtx_elem
                        pivot_found = True
                        A[[pivot, j]] = A[[j, pivot]] # Swap the row

                        for k in range(m):
                            if k != pivot:
                                A[k, :] = A[k, :] - A[pivot, :]*A[k, i]
                        break

        if pivot_found:
            pivot += 1 # Increment pivot to produce row echelon

    A = np.around(A, 2) # Limit precision to 2 decimal places
    return A

def get_rank_and_basis(A):
    
    # Get the RREF to determine linearly independent vectors
    rref_A = rref(A)
    
    # Get the mxn shape of the matrix
    m, n = A.shape

    independent_vec_idx = []

    # Get leading ones per row
    for i in range(m):
        for j in range(n):
            mtx_elem = rref_A[i, j]
            if float(mtx_elem) == 1.0:
                independent_vec_idx.append(j)
                break

    return len(independent_vec_idx), independent_vec_idx

def get_column_norm(Q):
    
    norm = []
    
    m, n = Q.shape
    
    for col in range(n):
        norm.append(np.linalg.norm(Q[:, col]))
        
    return norm

def get_normalized_null_space(mtx):
    
    m, n = mtx.shape
    
    if m < n:
        null_space = np.zeros(n)
    else:
        null_space = np.zeros(m)
    
    # Get the RREF to solve for null space
    mtx_rref = rref(mtx.copy())
    
    # Extract indices of independent vectors to be used as 
    # reference for calculations
    rank, independent_vecs = get_rank_and_basis(mtx.copy())
    
    pivot_found = False
    
    # Get the null space of the given matrix
    for col in range(n):
        
        if col in independent_vecs:
            continue
            
        for row in range(len(null_space)):
            if row not in independent_vecs and row <= (m-1):
                if not pivot_found:
                    null_space[row] = -1
                    pivot_found = True
                else:
                    null_space[row] = 0
            elif row in independent_vecs and row <= (m-1):
                null_space[row] = mtx_rref[row, col]
            else:
                null_space[row] = 0     
        print(f"null_space = {null_space}")
        null_space = null_space*1/np.linalg.norm(null_space) # Normalize the column vector
        break
    
    # Return a defensive copy to avoid issues
    return null_space.copy()
                                               
def qr_decomposition(A, epsilon=1e-7):

    # Get the mxn shape of the matrix
    m, n = A.shape
    
    # Initializes matrices Q and P
    if m < n:
        Q = np.zeros((m, m))
    else:
        Q = np.zeros((m, n))
    P = np.eye(n, n)
    
    pivot_count = 0
    
    # Base Case: Check first if A is the zero matrix
    # If zero matrix, Q = I_m, R is the zero matrix, and P = I_n
    if np.allclose(A, np.zeros((m, n), dtype="float"), rtol=1e-3, atol=epsilon):
        return np.identity(m), np.zeros((m, n)), np.identity(n)
    
    # Get the rank and basis to determine on which vectors to perform Gram-Schmidt 
    rank, independent_vec_idx = get_rank_and_basis(A.copy())

    # Perform Gram-Schmidt algorithm for getting orthonormal vectors
    for col in range(n):
        
        # Entries of Q only spans the column space of A
        if pivot_count >= m or pivot_count == rank:
            break
        
        # Perform column pivoting if column vector entries are all zeros
        # Swap with column with the highest absolute value of entries
        if np.linalg.norm(A[:, col]) < epsilon:
            
            # Get the column with the highest norm
            A_cols_norm = np.linalg.norm(np.absolute(A[:, col:]), axis=0)
            max_col_vec = np.max(A_cols_norm[col:])
            
            # If norm of column vector with max value is ~zero, stop performing Gram-Schmidt
            # Else if current vector has the highest norm, continue iteration
            if np.linalg.norm(max_col_vec) < epsilon:
                break
            elif np.linalg.norm(max_col_vec) == np.linalg.norm(A[:, col]):
                continue
            swap_idx = col+np.where(A_cols_norm == max_col_vec)[0][0]
            
            # Swap columns
            A[:, col], A[:, swap_idx] = A[:, swap_idx], A[:, col].copy()
            P[:, col], P[:, swap_idx] = P[:, swap_idx], P[:, col].copy()

        # Obtain projection vector p
        p = np.zeros(m)
        for i in range(pivot_count):
            dot_prod = np.dot(A[:, col], Q[:, i])
            p += dot_prod*Q[:, i]

        e = A[:, col] - p # Obtain orthogonal vector e = a - p
        
        # If e is the zero vector, skip
        if (np.allclose(e, np.zeros(m, dtype="float"), rtol=1e-3, atol=1e-5)):
            continue
        q = e*1/np.linalg.norm(e) # Normalize e to get q_i
        Q[:, pivot_count] = q
        pivot_count += 1

    # If Q is column deficient, get the null space of Q^T to obtain additional columns
    if pivot_count < Q.shape[1]:
        columns_to_add = Q.shape[1] - pivot_count
        for i in range(columns_to_add):
            null_space = get_normalized_null_space(np.transpose(Q))
            Q[:, pivot_count+i] = null_space

    R = np.transpose(Q) @ A

    return Q, R, P

def verify_qr(A, Q, R, P, suppress_success_flag=False):
    
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