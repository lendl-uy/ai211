# Code Author: Jan Lendl R. Uy
# Student Number: 2019-00312
# Class: AI 211 FZZQ

# Coding Challenge 3: QR Decomposition with Column Pivoting

import numpy as np
import time

EPSILON = 1e-5
NUM_TESTS = 10000

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

def get_basis(A):
    
    # Get the RREF to determine linearly independent vectors
    rref_A = rref(A)

    # Get the mxn shape of the matrix
    m, n = get_mtx_shape(A)

    leadingOnes = []

    # Get leading ones per row
    for i in range(m):
        for j in range(n):
            mtx_elem = A[i][j]
            if float(mtx_elem) == 1.0:
                leadingOnes.append(j)
                break

    return leadingOnes, len(leadingOnes)

# TODO: Finish implementing null space calculator
def get_null_space(A, Q, basis, independent_vecs):
    
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

    # Get the basis to determine which vectors to perform Gram-Schmidt 
    set_of_independent_vecs, basis = get_basis(A.copy())

    # If basis is less than number of rows, get the null space of A
    if basis < m:
        # Get the null space of A and update Q
        get_null_space(A.copy(), Q, basis, set_of_independent_vecs)

    # Perform Gram-Schmidt algorithm for getting orthonormal vectors
    for col in range(basis):
        
        # Skip the algorithm if current column is a dependent vector
        if col not in set_of_independent_vecs:
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
                # Obtain projection vector
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

def main():

    # Initialize matrix A to be decomposed
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]]).astype("float")
    '''
    A = np.array([[74, -80, 46, 33, 5, -65, 21, 93],
                  [18, -52, -70, -8, -60, -70, 0, 12],
                  [87, 59, -21, -33, 19, 38, -32, 82]]).astype("float")
    '''
    '''
    # Random number of rows for testing
    m = np.random.randint(3, 50)
    # Random number of cols for testing
    n = np.random.randint(3, 50)
    # Randomize entries of matrix A for testing
    A = np.random.randint(-100, 100, size=(m, n)).astype("float")
    '''
    # Pass a copy of A instead since Python will modify the original contents of A
    Q, R, P = qr_decomposition(A.copy(), EPSILON)

    print(f"--Below this line are the computed matrices of the decomposition--")
    print(f"A = {A}\n")
    print(f"P = {P}\n")
    print(f"Q = {Q}\n")
    print(f"R = {R}\n")

    # Verification of computed matrices
    AP = A @ P
    QR = Q @ R

    print(f"--Below this line is the verification of the decomposition--")
    print(f"AP = {AP}\n")
    print(f"QR = {QR}\n")

    # allclose() is used due to loss of precision when performing row-wise operations
    if np.allclose(np.subtract(AP, QR), np.zeros((3, 3), dtype="float"), rtol=1e-3, atol=1e-5):
        print("Decomposition is correct!")
    else:
        raise Exception("Something went wrong with the decomposition!")

    '''
    # Perform multiple tests and inspect runtime
    start = time.time()
    for n in range(NUM_TESTS):
        # Initialize matrix A to be decomposed
        # Random number of rows for testing
        m = np.random.randint(3, 50)
        # Random number of cols for testing
        n = np.random.randint(3, 50)
        # Randomize entries of matrix A for testing
        A = np.random.randint(-100, 100, size=(m, n)).astype("float")

        # Pass a copy of A instead since Python will modify the original contents of A
        Q, R, P = qr_decomposition(A.copy(), EPSILON)

        # Verification of computed Q, R, and P matrices
        QRP = np.round(Q @ R @ np.transpose(P))

        # allclose() is used due to loss of precision when performing row-wise operations
        if np.allclose(np.subtract(A, QRP), np.zeros((m, n), dtype="float"), rtol=1e-3, atol=1e-5):
            continue
            #print("Decomposition is correct!")
        else:
            print(f"A = {A}")
            print(f"QRP = {QRP}")

            print(f"A = {A}\n")
            print(f"P = {P}\n")
            print(f"Q = {Q}\n")
            print(f"R = {R}\n")
            raise Exception("Something went wrong with the decomposition!")
    
    print(f"All decomposition tests are correct!")
    print(f"Runtime for {NUM_TESTS} tests: {time.time()-start} s")
    '''
if __name__ == "__main__":
    main()