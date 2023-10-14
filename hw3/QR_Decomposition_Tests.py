# Code Author: Jan Lendl R. Uy
# Student Number: 2019-00312
# Class: AI 211 FZZQ

# Coding Challenge 3 Tests

from QR_Decomposition import *

def qr_decomposition_of_square_matrix_full_rank():
    
    print("---- Running QR decomposition test for square matrix with full rank ----")
    
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]]).astype("float")
    
    # Pass a copy of A instead since Python will modify the original contents of A
    Q, R, P = qr_decomposition(A.copy(), EPSILON)
    
    print("Results:")
    print(f"P = {P}\n")
    print(f"Q = {Q}\n")
    print(f"R = {R}\n")
    
    verify_qr(A, Q, R, P)
    
def qr_decomposition_of_square_matrix_rank_deficient():
    
    print("---- Running QR decomposition test for square matrix that is rank-deficient ----")
    
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]]).astype("float")
    
    # Pass a copy of A instead since Python will modify the original contents of A
    Q, R, P = qr_decomposition(A.copy(), EPSILON)
    
    print("Results:")
    print(f"P = {P}\n")
    print(f"Q = {Q}\n")
    print(f"R = {R}\n")
    
    verify_qr(A, Q, R, P)
    
def qr_decomposition_of_nonsquare_matrix_with_rows_greater_than_columns():
    
    print("---- Running QR decomposition test for non-square matrix where m > n ----")
    
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [10, 11, 12],
                  [13, 14, 15]]).astype("float")
    
    # Pass a copy of A instead since Python will modify the original contents of A
    Q, R, P = qr_decomposition(A.copy(), EPSILON)
    
    print("Results:")
    print(f"P = {P}\n")
    print(f"Q = {Q}\n")
    print(f"R = {R}\n")
    
    verify_qr(A, Q, R, P)
    
def qr_decomposition_of_nonsquare_matrix_with_columns_greater_than_rows():
    
    print("---- Running QR decomposition test for non-square matrix where n > m ----")
    
    A = np.array([[1, 3, 0, 0, 3],
                  [0, 0, 1, 0, 9],
                  [0, 0, 0, 1, -4]]).astype("float")
    
    # Pass a copy of A instead since Python will modify the original contents of A
    Q, R, P = qr_decomposition(A.copy(), EPSILON)
    
    print("Results:")
    print(f"P = {P}\n")
    print(f"Q = {Q}\n")
    print(f"R = {R}\n")
    
    # Verification of computed matrices
    AP = A @ P
    QR = Q @ R
    
    verify_qr(A, Q, R, P)
    
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

        verify_qr(A, Q, R, P)
        
    print(f"All decomposition tests are correct!")

def main():

    #qr_decomposition_of_square_matrix_full_rank()
    #qr_decomposition_of_square_matrix_rank_deficient()
    qr_decomposition_of_nonsquare_matrix_with_rows_greater_than_columns()
    #qr_decomposition_of_nonsquare_matrix_with_columns_greater_than_rows()
    #qr_decomposition_random_mtx(1000)

if __name__ == "__main__":
    main()