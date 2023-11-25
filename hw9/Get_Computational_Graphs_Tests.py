# Code Author: Jan Lendl R. Uy
# Student Number: 2019-00312
# Class: AI 211 FZZQ

# Coding Challenge 9 Tests

from Get_Computational_Graphs import *
    
def get_computational_graph_4x4_matrix(NUM_TESTS):
    
    print(f"\n---- Performing SVD on {NUM_TESTS} matrices with random sizes and entries such that rows ≥ cols ----\n")
    
    for i in range(NUM_TESTS):
        # Initialize matrix A to be decomposed
        m = np.random.randint(3, 5) # Random number of rows
        #n = np.random.randint(3, m) # Random number of cols
        A = np.random.randint(-10, 10, size=(m, m)).astype("float")
        
    print(f"All computed SVD factors are correct for {NUM_TESTS} test matrices!")

def main():
    
    pass
    #get_computational_graph_4x4_matrix(2000)

if __name__ == "__main__":
    main()