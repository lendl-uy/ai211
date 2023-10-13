# Code Author: Jan Lendl R. Uy
# Student Number: 2019-00312
# Class: AI 211 FZZQ

# Coding Challenge 3 Tests

from QR_Decomposition import *

def qr_decomposition_of_matrix_with_full_rank():
    
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
    
def qr_decomposition_of_matrix_with_rank_less_than_columns():
    
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
    
def qr_decomposition_of_matrix_with_rank_greater_than_columns():
    
    print("---- Running QR decomposition test for matrix with rank > columns ----")
    
    A = np.array([[1, 3, 0, 0, 3],
                  [0, 0, 1, 0, 9],
                  [0, 0, 0, 1, 4]]).astype("float")
    
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
    if np.allclose(np.subtract(AP, QR), np.zeros((3, 5), dtype="float"), rtol=1e-3, atol=1e-5):
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

    #qr_decomposition_of_matrix_with_full_rank()
    #qr_decomposition_of_matrix_with_rank_less_than_columns()
    qr_decomposition_of_matrix_with_rank_greater_than_columns()
    #qr_decomposition_random_mtx(1000)

if __name__ == "__main__":
    main()