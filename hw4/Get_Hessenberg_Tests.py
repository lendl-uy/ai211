# Code Author: Jan Lendl R. Uy
# Student Number: 2019-00312
# Class: AI 211 FZZQ

# Coding Challenge 4 Tests

from Get_Hessenberg import *
import scipy as sp

def compute_hessenberg_matrix_with_zero_columns():
    
    A = np.array([[1, 0, 0, 0],
                  [2, 0, 0, 0],
                  [3, 0, 0, 0],
                  [4, 0, 0, 0]]).astype("float")
    
    Hess = sp.linalg.hessenberg(A)
        
    Hess, H = get_hessenberg_matrix(A) # H is the Hessenberg matrix    
    
    print("Results:")
    print(f"Hess = {Hess}\n")
    print(f"H = {H}\n")
    
    verify_hessenberg_matrix(A, Hess, H)
    

def compute_hessenberg_matrix_of_zero_matrix():
    
    print("---- Computing the Hessenberg matrix of a zero matrix ----")
    
    A = np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]]).astype("float")
    
    Hess, H = get_hessenberg_matrix(A) # H is the Hessenberg matrix
    
    print("Results:")
    print(f"Hess = {Hess}\n")
    print(f"H = {H}\n")
    
    verify_hessenberg_matrix(A, Hess, H)

def compute_hessenberg_matrix_of_4x4_matrix():
    
    print("---- Computing the Hessenberg matrix of a 4x4 matrix ----")
    
    A = np.array([[1, 0, 2, 3],
                  [-1, 0, 5, 2],
                  [2, -2, 0, 0],
                  [2, -1, 2, 0]]).astype("float")
    
    Hess, H = get_hessenberg_matrix(A) # H is the Hessenberg matrix
    
    print("Results:")
    print(f"Hess = {Hess}\n")
    print(f"H = {H}\n")
    
    verify_hessenberg_matrix(A, Hess, H)
    
def compute_hessenberg_matrix_of_random_mtx_size(NUM_TESTS):
    
    print(f"---- Running {NUM_TESTS} tests for computing the Hessenberg matrix with random size and entries ----")
    
    for i in range(NUM_TESTS):
        # Initialize matrix A to be decomposed
        m = np.random.randint(3, 100) # Random number of rows
        A = np.random.randint(-100, 100, size=(m, m)).astype("float")

        # Pass a copy of A instead since Python will modify the original contents of A
        Hess, H = get_hessenberg_matrix(A)

        verify_hessenberg_matrix(A, Hess, H, suppress_success_flag=True)
        
    print(f"All decomposition tests are correct!")

def main():
    
    compute_hessenberg_matrix_with_zero_columns()
    compute_hessenberg_matrix_of_zero_matrix()
    compute_hessenberg_matrix_of_4x4_matrix()
    compute_hessenberg_matrix_of_random_mtx_size(100000)

if __name__ == "__main__":
    main()