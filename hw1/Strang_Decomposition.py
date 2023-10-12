# Code Author: Jan Lendl R. Uy
# Student Number: 2019-00312
# Class: AI 211 FZZQ

import numpy as np

def getMatrixShape(A):

    # Get the mxn shape of the matrix
    return A.shape[0], A.shape[1]

def getLeadingOnes(A):

    # Get the mxn shape of the matrix
    m, n = getMatrixShape(A)

    leadingOnes = []

    # Get leading ones per row
    for i in range(m):
        for j in range(n):
            mtx_elem = A[i][j]
            if float(mtx_elem) == 1.0:
                leadingOnes.append(j)
                break

    return leadingOnes, len(leadingOnes)

def rref(A):

    # Get the mxn shape of the matrix
    m, n = getMatrixShape(A)

    pivot = 0

    # RREF Algorithm
    # Look for the leftmost nonzero entry of each row
    # Perform row-wise operations to obtain a leading 1 and all entries in that
    # column to be zero
    for i in range(n):
        pivot_found = False # Reset pivot_found for every iteration of columns

        for j in range(m):
            mtx_elem = A[j, i]

            # Check for leftmost nonzero entry
            if mtx_elem == 0:
                continue
            else:

                # Perform row-wise operations to convert matrix into its RREF
                if j == pivot:
                    A[j, :] *= 1/mtx_elem
                    pivot_found = True

                    for k in range(m):
                        if k != pivot:
                            A[k, :] = A[k, :] - A[pivot, :]*A[k, i]
                    break
                else:

                    # Swap rows with the topmost row without a pivot yet
                    if not pivot_found and j>pivot:
                        A[j, :] *= 1/mtx_elem
                        pivot_found = True
                        A[[pivot, j]] = A[[j, pivot]] # Swap the row

                        for k in range(m):
                            if k != pivot:
                                A[k, :] = A[k, :] - A[pivot, :]*A[k, i]
                        break

        if pivot_found:
            pivot += 1 # Increment pivot to produce row echelon

    A = np.around(A, 2) # Limit precision to 2 decimal places
    return A

def crDecomposition(A):

    A_copy = A.copy()

    # Matrix R: Get RREF of A
    R = rref(A_copy)
    leading, rank = getLeadingOnes(R) # Get indices of leading 1s and rank of R
    R = R[:rank,] # Truncate R to remove zero row/s

    # Matric C: Get independent columns of A
    C = np.array([A[:, leading.pop(0)]]).astype("float")
    for col in leading:
        ind_col = [np.ndarray.tolist(A[:, col])]
        C = np.concatenate((C, ind_col))

    # Since retrieval of columns follow a row format, get the transpose of C
    C = np.transpose(C)

    return C, R

def main():

    # Initialize matrix A to be decomposed
    m = np.random.randint(3, 10) # Random number of rows for testing
    n = np.random.randint(3, 10) # Random number of cols for testing
    A = np.random.randint(100, size=(m, n)).astype("float") # Randomize entries of matrix A for testing

    C, R = crDecomposition(A)

    print(f"A = {A}\n")
    print(f"C = {C}\n")
    print(f"R = {R}\n")
    print(f"For Checking: A = {np.matmul(C, R)}")

    # allclose() is used due to loss of precision when performing RREF
    # Tolerance is set to ones digit
    if np.allclose(A, np.matmul(C, R), 1e0):
        print("Decomposition is correct!")

if __name__ == "__main__":
    main()