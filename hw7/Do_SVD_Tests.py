# Code Author: Jan Lendl R. Uy
# Student Number: 2019-00312
# Class: AI 211 FZZQ

# Coding Challenge 7 Tests

from Do_SVD import *

def perform_svd_on_4x4_dense_matrix():
    
    print(f"\n---- Performing SVD on a 4x4 bidiagonal matrix ----\n")
    
    A = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]]).astype("float")

    print(f"A = {A}")
    
    U, Σ, Vt = do_svd_via_golub_reinsch(A)
    
    print("\nResults:")
    print(f"U = {U}\n")
    print(f"Σ = {Σ}\n")
    print(f"Vt = {Vt}\n")
    
    verify_svd(A, U, Σ, Vt)
        
def perform_svd_on_4x4_bidiagonal_matrix():
    
    print(f"\n---- Performing SVD on a 4x4 bidiagonal matrix ----\n")
    
    A = np.array([[1, 1, 0, 0],
                  [0, 2, 1, 0],
                  [0, 0, 3, 1],
                  [0, 0, 0, 4]]).astype("float")

    print(f"A = {A}")
    
    U, Σ, Vt = do_svd_via_golub_reinsch(A)
    
    print("\nResults:")
    print(f"U = {U}\n")
    print(f"Σ = {Σ}\n")
    print(f"Vt = {Vt}\n")
    
    verify_svd(A, U, Σ, Vt)
    
def perform_svd_on_matrix_with_random_sizes_and_entries(NUM_TESTS):
    
    print(f"\n---- Performing SVD on {NUM_TESTS} matrices with random sizes and entries such that rows ≥ cols ----\n")
    
    for i in range(NUM_TESTS):
        # Initialize matrix A to be decomposed
        m = np.random.randint(3, 6) # Random number of rows
        n = np.random.randint(m, m+3) # Random number of cols
        A = np.random.randint(-10, 10, size=(m, n)).astype("float")
        
        #print(f"A = {A}")
        U, Σ, Vt = do_svd_via_golub_reinsch(A)
        verify_svd(A, U, Σ, Vt, suppress_success_flag=False)
        
    print(f"All computed SVD factors are correct for {NUM_TESTS} test matrices!")

def main():
    
    perform_svd_on_4x4_dense_matrix()
    perform_svd_on_4x4_bidiagonal_matrix()
    perform_svd_on_matrix_with_random_sizes_and_entries(5)

if __name__ == "__main__":
    main()