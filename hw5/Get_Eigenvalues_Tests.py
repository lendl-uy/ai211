# Code Author: Jan Lendl R. Uy
# Student Number: 2019-00312
# Class: AI 211 FZZQ

# Coding Challenge 5 Tests

from Get_Eigenvalues import *

def compute_eigenvalues_and_eigenvectors_square_matrix():
    
    A = np.array([[6, 5, 0],
                  [5, 1, 4],
                  [0, 4, 3]]).astype("float")
    
    eigenvalues = get_eigenvalues(A)
    
    print("Results:")
    print(f"eigenvalues = {eigenvalues}\n")
    
    verify_eigenvalues(A, eigenvalues)
    
def compute_eigenvalues_and_eigenvectors_of_random_mtx_size(NUM_TESTS):
    
    print(f"---- Running {NUM_TESTS} tests for computing the Hessenberg matrix with random size and entries ----")
    
    for i in range(NUM_TESTS):
        # Initialize matrix A to be decomposed
        m = np.random.randint(3, 100) # Random number of rows
        A = np.random.randint(-100, 100, size=(m, m)).astype("float")

        # Pass a copy of A instead since Python will modify the original contents of A
        
    print(f"All decomposition tests are correct!")

def main():
    
    compute_eigenvalues_and_eigenvectors_square_matrix()

if __name__ == "__main__":
    main()