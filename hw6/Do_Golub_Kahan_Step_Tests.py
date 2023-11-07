# Code Author: Jan Lendl R. Uy
# Student Number: 2019-00312
# Class: AI 211 FZZQ

# Coding Challenge 6 Tests

from Do_Golub_Kahan_Step import *

def perform_bidiagonalization_3x3_matrix():
    
    print(f"\n---- Performing Golub-Kahan bidiagonalization on a 3x3 matrix ----\n")

    A = np.array([[6, 5, 0],
                  [5, 1, 4],
                  [0, 4, 3]]).astype("float")

    print(f"A = {A}\n")
    
    B = get_bidiagonal_matrix(A)
    
    print("Results:\n")
    print(f"B = {B}\n")
    
    verify_bidiagonal_matrix(A, B)
        
def perform_tridiagonalization_3x3_matrix():
    
    print(f"\n---- Performing tridiagonalization on a 3x3 matrix ----\n")
    
    A = np.array([[6, 5, 0],
                  [5, 1, 4],
                  [0, 4, 3]]).astype("float")

    print(f"A = {A}\n")
        
    T = get_tridiagonal_matrix(A)
    
    print("Results:\n")
    print(f"T = {T}\n")
    
    verify_tridiagonal_matrix(A, T)
    
def perform_bidiagonalization_on_matrix_with_random_sizes_and_entries(NUM_TESTS):
    
    print(f"\n---- Performing {NUM_TESTS} bidiagonalization tests on matrices with random sizes and entries ----\n")
    
    for i in range(NUM_TESTS):
        # Initialize matrix A to be decomposed
        m = np.random.randint(3, 50) # Random number of rows
        n = np.random.randint(3, 50) # Random number of cols
        A = np.random.randint(-10, 10, size=(m, n)).astype("float")
        
        #print(f"A = {A}")
        
        B = get_bidiagonal_matrix(A)
        
        verify_bidiagonal_matrix(A, B, suppress_success_flag=True)
        
    print(f"All computed bidiagonal matrices are correct for {NUM_TESTS} test matrices!")
    
def perform_tridiagonalization_on_matrix_with_random_sizes_and_entries(NUM_TESTS):
    
    print(f"\n---- Performing {NUM_TESTS} tridiagonalization tests on matrices with random sizes and entries ----\n")
    
    for i in range(NUM_TESTS):
        # Initialize matrix A to be decomposed
        m = np.random.randint(3, 50) # Random number of rows
        n = np.random.randint(3, 50) # Random number of cols
        A = np.random.randint(-10, 10, size=(m, n)).astype("float")
        
        #print(f"A = {A}")
        
        T = get_tridiagonal_matrix(A)
        
        verify_tridiagonal_matrix(A, T, suppress_success_flag=True)

    print(f"All computed tridiagonal matrices are correct for {NUM_TESTS} test matrices!")

def main():
    
    perform_bidiagonalization_3x3_matrix()
    perform_tridiagonalization_3x3_matrix()
    perform_bidiagonalization_on_matrix_with_random_sizes_and_entries(100000)
    perform_tridiagonalization_on_matrix_with_random_sizes_and_entries(100000)

if __name__ == "__main__":
    main()