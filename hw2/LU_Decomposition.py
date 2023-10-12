# Code Author: Jan Lendl R. Uy
# Student Number: 2019-00312
# Class: AI 211 FZZQ
# Coding Challenge 2

import numpy as np

EPSILON = 1e-8

def get_mtx_shape(A):

    # Get the mxn shape of the matrix
    return A.shape[0], A.shape[1]

def is_square(A):

    # Get the mxn shape of the matrix
    m, n = get_mtx_shape(A)

    return m == n

def get_determinant(A):

    if not is_square(A):
        raise Exception("The matrix is not a square! The determinant of A will not be computed.")

    row_swaps, L, U = plu_decomposition(A.copy(), get_det=True)

    # det(A) = det(P^-1) * det(L) * det(U)
    determinant_L = np.prod(np.diag(L))
    determinant_U = np.prod(np.diag(U))

    # det(P) = -1^(permutations or row swaps)
    determinant_P = (-1)**row_swaps

    determinant_A = determinant_P*determinant_L*determinant_U

    return determinant_A

def plu_decomposition(A, e=1e-10, get_det=False):

    # Get the mxn shape of the matrix
    m, n = get_mtx_shape(A)

    # Declare initial values for mxm matrices P and L
    P = np.identity(m)
    L = np.identity(m)

    # Keep track of row swaps if determinant will be computed since det(P) = (-1)^(no. of row swaps)
    row_swaps = 0

    # Perform Gaussian elimination on entries below the diagonal
    # Add the coefficient used for each row-wise operation in the correct entry of L
    # Update P, L, and U (i.e. current A) when row-swapping occurs
    for col in range(n-1):
        for row in range(m-1):
            mtx_elem = A[row, col]

            if row == col:

                # Pivot found if > epsilon. Perform row-wise operations to make entries below the pivot zero.
                if abs(mtx_elem) > e:
                    pivot = mtx_elem
                    for i in range(row+1, m):
                        if abs(A[i, col]) > e:
                            coeff = A[i, col]/pivot
                            A[i, :] = A[i, :] - A[row, :]*(coeff)
                            L[i, col] = coeff # Saves the coefficient used in the operation to L

                # Else, perform row-swapping for matrices P, L, and U (i.e. current A) if element < epsilon
                # For numerical stability, swap with row containing highest absolute value at the current column iteration
                else:

                    row_swaps += 1

                    pivot_col = np.ndarray.tolist(np.absolute(A[:, col].copy()))
                    max_elem_col = max(pivot_col)
                    max_elem_idx = pivot_col.index(max_elem_col)
                    if max_elem_idx < row:
                        continue

                    A[[row, max_elem_idx]] = A[[max_elem_idx, row]]
                    P[[row, max_elem_idx]] = P[[max_elem_idx, row]]

                    # For L, swapping only occurs for slices of the matrix below the diagonal
                    tmp = L[row, :col].copy()
                    L[row, :col] = L[max_elem_idx, :col]
                    L[max_elem_idx, :col] = tmp

    U = A
    
    if get_det:
        return row_swaps, L, U
    return P, L, U

def main():

    # Initialize matrix A to be decomposed
    m = np.random.randint(3, 50) # Random number of rows for testing
    n = np.random.randint(3, 50) # Random number of cols for testing
    A = np.random.randint(-100, 100, size=(m, n)).astype("float") # Randomize entries of matrix A for testing
    
    # Pass a copy of A instead since Python will modify the original contents of A
    P, L, U = plu_decomposition(A.copy(), EPSILON)

    print(f"--Below this line are the computed matrices of the decomposition--")
    print(f"P = {P}\n")
    print(f"A = {A}\n")
    print(f"L = {L}\n")
    print(f"U = {U}\n")

    # Get the determinant of A if it is square
    if is_square(A):
        determinant_A = get_determinant(A)
        print(f"det(A) = {determinant_A}\n")
        print(f"det(A) from numpy = {np.linalg.det(A)}\n")

    # Verification of computed P, L, and U matrices
    PLU = np.round(np.linalg.inv(P) @ L @ U)

    print(f"--Below this line is the verification of the decomposition--")
    print(f"A = {A}\n")
    print(f"PLU = {PLU}\n")

    # allclose() is used due to loss of precision when performing Gaussian Elimination
    if np.allclose(np.subtract(A, PLU), np.zeros((m, n), dtype="float"), rtol=1e-3, atol=1e-5):
        print("Decomposition is correct!")
    else:
        raise Exception("Something went wrong with the decomposition!")
    
if __name__ == "__main__":
    main()