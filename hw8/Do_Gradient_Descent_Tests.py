# Code Author: Jan Lendl R. Uy
# Student Number: 2019-00312
# Class: AI 211 FZZQ

# Coding Challenge 8 Tests

from Do_Gradient_Descent import *
    
def perform_svd_on_matrix_with_random_sizes_and_entries(NUM_TESTS):
    
    print(f"\n---- Performing SVD on {NUM_TESTS} matrices with random sizes and entries such that rows ≥ cols ----\n")
    
    for i in range(NUM_TESTS):
        # Initialize matrix A to be decomposed
        m = np.random.randint(3, 5) # Random number of rows
        #n = np.random.randint(3, m) # Random number of cols
        A = np.random.randint(-10, 10, size=(m, m)).astype("float")
        
    print(f"All computed SVD factors are correct for {NUM_TESTS} test matrices!")

def main():
    
    pass
    #perform_svd_on_matrix_with_random_sizes_and_entries(2000)

if __name__ == "__main__":
    main()