# Code Author: Jan Lendl R. Uy
# Student Number: 2019-00312
# Class: AI 211 FZZQ
# Coding Challenge 3

import numpy as np
import time

EPSILON = 1e-5
NUM_TESTS = 10000

def get_mtx_shape(A):

    # Get the mxn shape of the matrix
    return A.shape[0], A.shape[1]

def qr_decomposition(A, epsilon=1e-8):

    # Get the mxn shape of the matrix
    m, n = get_mtx_shape(A)

    # Declare initial values for mxm matrices P and L
    Q = np.zeros((m, m))
    R = np.zeros((m, n))
    P = np.eye(n, n)

    pivot_count = 0

    for col in range(n):

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
    '''
    A = np.array([[74, -80, 46, 33, 5, -65, 21, 93],
                  [18, -52, -70, -8, -60, -70, 0, 12],
                  [87, 59, -21, -33, 19, 38, -32, 82]]).astype("float")
    '''
    # Random number of rows for testing
    m = np.random.randint(3, 50)
    # Random number of cols for testing
    n = np.random.randint(3, 50)
    # Randomize entries of matrix A for testing
    A = np.random.randint(-100, 100, size=(m, n)).astype("float")

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
    if np.allclose(np.subtract(AP, QR), np.zeros((m, n), dtype="float"), rtol=1e-3, atol=1e-5):
        print("Decomposition is correct!")
    else:
        raise Exception("Something went wrong with the decomposition!")

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

    print(f"Runtime for {NUM_TESTS} tests: {time.time()-start} s")

if __name__ == "__main__":
    main()