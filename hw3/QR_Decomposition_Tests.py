# Code Author: Jan Lendl R. Uy
# Student Number: 2019-00312
# Class: AI 211 FZZQ

# Coding Challenge 3 Tests

import numpy as np

from QR_Decomposition import Matrix

def qr_decomposition_of_matrix_will_full_rank():
    
    mtx = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]]).astype("float")
    
    A = Matrix(mtx)
    
    # Pass a copy of A instead since Python will modify the original contents of A
    Q, R, P = A.qr(A.copy(), EPSILON)
    
def qr_decomposition_of_matrix_will_rank_less_than_columns(A):
    
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]]).astype("float")
    
    # Pass a copy of A instead since Python will modify the original contents of A
    Q, R, P = qr_decomposition(A.copy(), EPSILON)
    
    # Verification of computed matrices
    AP = A @ P
    QR = Q @ R
    
def main():

    # Initialize matrix A to be decomposed
    
    # Test 1: 
    
    # Test 2: 
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