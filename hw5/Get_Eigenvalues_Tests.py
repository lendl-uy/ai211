# Code Author: Jan Lendl R. Uy
# Student Number: 2019-00312
# Class: AI 211 FZZQ

# Coding Challenge 5 Tests

from Get_Eigenvalues import *

def compute_eigenvalues_and_eigenvectors_for_symmetric_matrix_where_standard_qr_fails():
    
    print(f"\n---- Running eigenvalue and eigenvector test for a matrix where standard QR algorithm fails ----\n")
    
    A = np.array([[0, 1],
                  [1, 0]]).astype("float")
    
    print(f"A = {A}")
    
    eigenvalues, eigenvectors = get_eigenvalues(A)
    
    print("\nResults:")
    print(f"eigenvalues = {eigenvalues}\n")
    print(f"eigenvectors = {eigenvectors}\n")
    
    verify_eigenvalues(A, eigenvalues)
    verify_eigenvectors(A, eigenvalues, eigenvectors)

def compute_eigenvalues_and_eigenvectors_symmetric_matrix():
    
    print(f"\n---- Running eigenvalue and eigenvector test for a symmetric 3x3 matrix ----\n")
    
    A = np.array([[6, 5, 0],
                  [5, 1, 4],
                  [0, 4, 3]]).astype("float")

    print(f"A = {A}")
    
    eigenvalues, eigenvectors = get_eigenvalues(A)
    
    print("\nResults:")
    print(f"eigenvalues = {eigenvalues}\n")
    print(f"eigenvectors = {eigenvectors}\n")
    
    verify_eigenvalues(A, eigenvalues)
    verify_eigenvectors(A, eigenvalues, eigenvectors)
    
def compute_eigenvalues_and_eigenvectors_of_symmetric_matrices_with_random_entries(NUM_TESTS):
    
    print(f"\n---- Running {NUM_TESTS} tests for computing the eigenvalues and eigenvectors of a symmetric matrix with random entries ----\n")
    
    for i in range(NUM_TESTS):
        # Initialize matrix A to be decomposed
        m = np.random.randint(3, 50) # Random number of rows
        A = np.random.randint(-10, 10, size=(m, m)).astype("float")
        A = (A+A.T)/2 # Makes the matrix symmetric
        
        #print(f"A = {A}")
        
        # Perform QR algorithm
        # Return eigenvalues and eigenvectors if A is symmetric, 
        # else just return eigenvalues
        if is_symmetric(A):
            eigenvalues, eigenvectors = get_eigenvalues(A)
            verify_eigenvalues(A, eigenvalues, suppress_success_flag=True)
            verify_eigenvectors(A, eigenvalues, eigenvectors, suppress_success_flag=True)
        else:
            eigenvalues = get_eigenvalues(A)
            verify_eigenvalues(A, eigenvalues, suppress_success_flag=True)
        
    print(f"All eigenvalues and eigenvectors are correct for {NUM_TESTS} test matrices!")

def main():
    
    compute_eigenvalues_and_eigenvectors_for_symmetric_matrix_where_standard_qr_fails()
    compute_eigenvalues_and_eigenvectors_symmetric_matrix()
    compute_eigenvalues_and_eigenvectors_of_symmetric_matrices_with_random_entries(10000)

if __name__ == "__main__":
    main()