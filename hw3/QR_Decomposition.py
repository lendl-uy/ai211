# Code Author: Jan Lendl R. Uy
# Student Number: 2019-00312
# Class: AI 211 FZZQ

# Coding Challenge 3: QR Decomposition with Column Pivoting

import numpy as np
import time

EPSILON = 1e-5

def get_mtx_shape(A):

    # Get the mxn shape of the matrix
    return A.shape[0], A.shape[1]

def rref(A):

    # Get the mxn shape of the matrix
    m, n = get_mtx_shape(A)

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

def get_rank(A):
    
    # Get the RREF to determine linearly independent vectors
    rref_A = rref(A)

    # Get the mxn shape of the matrix
    m, n = get_mtx_shape(A)

    independent_vec_idx = []

    # Get leading ones per row
    for i in range(m):
        for j in range(n):
            mtx_elem = rref_A[i, j]
            if float(mtx_elem) == 1.0:
                independent_vec_idx.append(j)
                break

    return len(independent_vec_idx), independent_vec_idx

# TODO: Finish implementing null space calculator
def get_null_space(A, Q, rank, independent_vecs):
    
    # Get the RREF to determine linearly independent vectors
    rref_A = rref(A)

    m, n = get_mtx_shape(A)
                
    # Get the null space based on the basis and the index of 
    # independent vectors
    for col in range(n):
        
        if col not in independent_vecs:
            
            for row in range(n):
                if row not in independent_vecs:
                    Q[row, col] = -1
                elif row in independent_vecs and row <= col:
                    Q[row, col] = rref_A[row, col]
                else:
                    Q[row, col] = 0
                                            
def qr_decomposition(A, epsilon=1e-8):

    # Get the mxn shape of the matrix
    m, n = get_mtx_shape(A)
    
    # Initialize matrices Q, R, and P
    Q = np.zeros((m, m))
    R = np.zeros((m, n))
    P = np.eye(n, n)
    
    pivot_count = 0

    # Get the rank and columns of independent vectors to determine 
    # which vectors to perform Gram-Schmidt 
    rank, independent_vec_idx = get_rank(A.copy())

    # If rank is less than number of rows, get the null space of A
    if rank < m:
        
        # Get the null space of A and update Q
        get_null_space(A.copy(), Q, rank, independent_vec_idx)

    # Perform Gram-Schmidt algorithm for getting orthonormal vectors
    for col in range(rank):
        
        # Skip the algorithm if current column is a dependent vector
        if col not in independent_vec_idx:
            continue

        # Entries of Q only spans the column space of A
        if col >= m:
            break

        for row in range(m):

            mtx_elem = A[row, col]

            # Perform column pivoting if entry is less than specified epsilon
            if abs(mtx_elem) < epsilon:

                col_vec = np.absolute(A[row, col:])
                swap_idx = col+np.where(col_vec == np.max(col_vec))[0][0]
                A[:, col], A[:, swap_idx] = A[:, swap_idx], A[:, col].copy()
                P[:, col], P[:, swap_idx] = P[:, swap_idx], P[:, col].copy()

            if abs(mtx_elem) > epsilon:
                # Obtain projection vector p
                p = np.zeros(m)
                for i in range(pivot_count):
                    dot_prod = np.dot(A[:, pivot_count], Q[:, i])
                    p += dot_prod*Q[:, i]

                e = A[:, pivot_count] - p # Obtain orthogonal vector e = a - p
                q = e*1/np.linalg.norm(e) # Normalize e to get q_i
                Q[:, pivot_count] = q
                pivot_count += 1
                break

    R = np.transpose(Q) @ A

    return Q, R, P

def qr_decomposition_of_matrix_will_full_rank():
    
    print("---- Running QR decomposition test for matrix with full rank ----")
    
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]]).astype("float")
    
    # Pass a copy of A instead since Python will modify the original contents of A
    Q, R, P = qr_decomposition(A.copy(), EPSILON)
    
    print("Results:")
    print(f"P = {P}\n")
    print(f"Q = {Q}\n")
    print(f"R = {R}\n")
    
    # Verification of computed matrices
    AP = A @ P
    QR = Q @ R
    
    # allclose() is used due to loss of precision when performing row-wise operations
    if np.allclose(np.subtract(AP, QR), np.zeros((3, 3), dtype="float"), rtol=1e-3, atol=1e-5):
        print("Decomposition is correct!")
    else:
        raise RuntimeError("Something went wrong with the decomposition!")
    
def qr_decomposition_of_matrix_will_rank_less_than_columns():
    
    print("---- Running QR decomposition test for matrix with rank < columns ----")
    
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]]).astype("float")
    
    # Pass a copy of A instead since Python will modify the original contents of A
    Q, R, P = qr_decomposition(A.copy(), EPSILON)
    
    print("Results:")
    print(f"P = {P}\n")
    print(f"Q = {Q}\n")
    print(f"R = {R}\n")
    
    # Verification of computed matrices
    AP = A @ P
    QR = Q @ R
    
    # allclose() is used due to loss of precision when performing row-wise operations
    if np.allclose(np.subtract(AP, QR), np.zeros((3, 3), dtype="float"), rtol=1e-3, atol=1e-5):
        print("Decomposition is correct!")
    else:
        raise RuntimeError("Something went wrong with the decomposition!")
    
def qr_decomposition_random_mtx(NUM_TESTS):
    
    print("---- Running QR decomposition tests for random matrix sizes and random entries ----")
    
    for n in range(NUM_TESTS):
        # Initialize matrix A to be decomposed
        m = np.random.randint(3, 50) # Random number of rows
        n = np.random.randint(3, 50) # Random number of columns
        A = np.random.randint(-100, 100, size=(m, n)).astype("float")

        # Pass a copy of A instead since Python will modify the original contents of A
        Q, R, P = qr_decomposition(A.copy(), EPSILON)

        # Verification of computed matrices
        AP = A @ P
        QR = Q @ R

        # allclose() is used due to loss of precision when performing row-wise operations
        if np.allclose(np.subtract(AP, QR), np.zeros((m, n), dtype="float"), rtol=1e-3, atol=1e-5):
            #print("Decomposition is correct!")
            continue
        else:
            raise RuntimeError("Something went wrong with the decomposition!")
        
    print(f"All decomposition tests are correct!")

def main():

    qr_decomposition_of_matrix_will_full_rank()
    qr_decomposition_of_matrix_will_rank_less_than_columns()
    qr_decomposition_random_mtx(1000)

if __name__ == "__main__":
    main()