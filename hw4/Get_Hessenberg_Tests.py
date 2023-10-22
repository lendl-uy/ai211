# Code Author: Jan Lendl R. Uy
# Student Number: 2019-00312
# Class: AI 211 FZZQ

# Coding Challenge 4 Tests

from Get_Hessenberg import *

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

def main():
    
    compute_hessenberg_matrix_of_4x4_matrix()

if __name__ == "__main__":
    main()