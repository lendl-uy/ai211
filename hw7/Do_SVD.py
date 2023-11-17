# Code Author: Jan Lendl R. Uy
# Student Number: 2019-00312
# Class: AI 211 FZZQ

# Coding Challenge 7: Perform SVD

# Useful references:
# [1] https://drsfenner.org/blog/2016/03/householder-bidiagonalization/

import numpy as np
import math

EPSILON = 1e-10
MAX_ITERATION_COUNT = 20000

np.set_printoptions(precision=2)

def is_diagonal(A):
    
    rows, cols = A.shape
    
    B = get_diagonal_copy(A)
    
    if not np.allclose(B-A, np.zeros((rows, cols), dtype="float"), 
                    rtol=1e-3, atol=1e-3):
        return False
    return True 

def get_diagonal_copy(A):
    
    rows, cols = A.shape
    
    D = np.zeros((rows, cols))
    
    for i in range(rows):
        for j in range(cols):
            if i == j:
                D[i, j] = A[i, j]
                
    return D

def get_bidiagonal_copy(A):
    
    rows, cols = A.shape
    
    B = np.zeros((rows, cols))
    
    for i in range(rows):
        for j in range(cols):
            if j >= i and j < i+2:
                B[i, j] = A[i, j]
                
    return B

def zero_out_small_numbers(A, tol=EPSILON):
    
    rows, cols = A.shape
    
    for i in range(rows):
        for j in range(cols):
            mtx_elem = A[i, j]
            if abs(mtx_elem) < EPSILON:
                mtx_elem = A[i, j] = 0.0
                
def get_householder_vector(x, i):
    
    e = np.zeros(x.size)

    if abs(x[0]) < EPSILON:
        e[0] = 1.0
        u = x + np.linalg.norm(x)*e
    else:
        e[0] = np.sign(x[0])
        u = x + np.sign(x[0])*np.linalg.norm(x)*e
            
    # Retain u if it is the ~zero vector
    if np.sum(np.absolute(u)) > EPSILON:
        u *= 1/np.linalg.norm(u)
    u = u[:][np.newaxis].T # Cast 1D array to a column vector of a 2D array
    
    return u
           
def get_householder_matrix(house, size, i):
    # house is the Householder reflector/vector
    House = np.identity(size)
    House[i:, i:] -= 2 * house @ house.T
    
    return House

def get_wilkinson_shift(A, block_pos=0):
    
    rows, cols = A.shape # Get the number of rows and columns of A
    
    # Extract the 2x2 block matrix B at the bottom left of the matrix
    # Compute "estimated eigenvalue" (mu) from B
    B = A[rows-2-block_pos: rows-block_pos, cols-2-block_pos: cols-block_pos]
    #print(f"B = {B}")
    a = B[0, 0]
    b_10 = B[1, 0]
    b_01 = B[0, 1]
    c = B[1, 1]
    l = (a-c)/2
    
    # If l = 0, set sgn(l) = 1
    if l == 0.0:
        µ = c - (b_10*b_01)/(abs(l) + math.hypot(l, b_10*b_01)) # Shift coefficient
    else:
        µ = c - (np.sign(l)*b_10*b_01)/(abs(l) + math.hypot(l, b_10*b_01)) # Shift coefficient

    return µ

def givens_rotation(A, i, j):
    
    rows, cols = A.shape # Get the number of rows and columns of A
    
    G = np.eye(rows, cols) # Initialize the Givens rotation matrix
    
    # Set the 2x2 elements at which rotation will be performed
    a = A[j, j]
    b = A[i, j]
    r = math.hypot(a, b)
    
    #print(f"rotate: (a,b) -> ({a}, {b})")
    #print(f"r = {r}")
    
    G[i, i], G[j, j] = a/r, a/r
    G[i, j] = -b/r
    G[j, i] = b/r

    return G

def get_eigenvalues_2x2(B):
        
    a = B[0, 0]
    b = B[0, 1]
    c = B[1, 0]
    d = B[1, 1]
    
    # Eigenvalue formula for 2x2 matrix
    λ_1 = ((a+d) + math.sqrt((a+d)**2 - 4*(a*d-b*c)))/2
    λ_2 = ((a+d) - math.sqrt((a+d)**2 - 4*(a*d-b*c)))/2

    return λ_1, λ_2
       
def get_bidiagonal_matrix(A):
    
    rows, cols = A.shape
    
    B = A.copy()
    
    U = np.identity(rows)
    Vt = np.identity(cols)
    
    # Check if input matrix is already bidiagonal
    diff = B-get_bidiagonal_copy(A)
    if np.allclose(diff, np.zeros((rows, cols))):
        return U, B, Vt
    
    # Golub-Kahan Bidiagonalization
    # Essentially performing Householder transformation for each column AND
    # row of the input matrix to achieve bidiagonal form
    if cols > rows:
        m = rows
    else:
        m = cols
        
    for i in range(m):
        
        # Column-wise Householder transformation
        if i < rows-1:
            x = B[i:, i]
            u = get_householder_vector(x, i)
            B[i:, i:] -= 2*u @ (u.T @ B[i:, i:])
            Q = get_householder_matrix(u, rows, i)
            U = U @ Q

        # Row-wise Householder transformation
        if i < cols-2:
            x = B[i, i+1:]
            v = get_householder_vector(x, i)
            B[i:, i+1:] -= 2*(B[i:, i+1:] @ v) @ v.T
            P = get_householder_matrix(v, cols, i+1)
            Vt = P @ Vt
                  
    zero_out_small_numbers(B)

    return U, B, Vt

def get_tridiagonal_matrix(B):
        
    rows, cols = B.shape
    
    # Generate a larger matrix H such that H = {[0 B.T], [B 0]}
    # Permuting H properly will result in a tridiagonal matrix
    H = np.zeros((rows+cols, rows+cols))
    
    H[:cols, cols:] = B.T # Set upper right submatrix to B.T
    H[cols:, :cols] = B # Set bottom left submatrix to B
    
    # H has zero rows and/or columns when rows > cols
    # Initialize iteration var m to 2*cols < rows+cols to avoid permuting
    # these zero rows and/or columns
    if rows > cols:
        m = 2*cols
        H = H[:m, :m]
    else:
        m = rows+cols
        
    # Initialize permutation matrix P that shifts columns to achieve tridiagonal form
    # whose permutations are determined by the following piecewise equation:
    # pi = i/2, i mod 2 == 0
    # pi = floor(i/2) + ceil((rows+cols)/2), i mod 2 == 1
    P = np.zeros((m, m))
    
    for i in range(0, m):
        if i%2 == 0:
            pi = i//2 # Permutation index for even columns, // used for int division
            P[pi, i] = 1.0
        else:
            pi = i//2 + math.ceil((m)/2) # Permutation index for odd columns
            P[pi, i] = 1.0
        
    T = P.T @ H @ P
    
    return T

def golub_kahan_step(U, B, V, n, p, q):
        
    T = get_tridiagonal_matrix(B)
    B_22 = B[p:n-q, p:n-q]
    #T = B_22.T @ B_22
    
    rows, cols = T.shape
    
    C = T[rows-2:, cols-2:]
    #print(f"B_22 = {B_22}")
    #print(f"block = {C}")
    λ_1, λ_2 = get_eigenvalues_2x2(C)
    if abs(λ_1-C[1,1]) <= abs(λ_2-C[1,1]):
        µ = λ_1
    else:
        µ = λ_2
    #µ = get_wilkinson_shift(T)
        
    alpha = B[p, p]**2 - µ
    beta = B[p, p]*B[p, p+1]
    #B[p, p] = alpha
    #B[p, p+1] = beta

    for i in range(p, n-q):
        G = givens_rotation(B.T, i+1, i)
        B = B @ G.T
        V = V @ G.T
                
        #alpha = B[i, i]
        #beta = B[i+1, i]
        
        G = givens_rotation(B, i+1, i)
        B = G @ B
        U = U @ G.T
        
        #if i < n-q-1:
            #alpha = B[i, i+1]
            #beta = B[i, i+2]
        
    return U, B, V
        
def do_svd_via_golub_reinsch(A):
    
    U, B, Vt = get_bidiagonal_matrix(A)
    #print(f"U @ B @ Vt = {U @ B @ Vt}") # Sanity check, must equal original matrix A

    rows, cols = B.shape
    #print(f"B = {B}")
    Σ = B.copy()
    V = Vt.T
    
    q = 1
    p = 0
    iteration_count = 0
    
    while q < cols:
                
        # Zero out superdiagonal entry if 
        # |b[i,i+1]| <= EPSILON(|b[i,i]|+|b[i+1,i+1]|))
        for j in range(0, cols-1):
            if abs(B[j, j+1]) <= EPSILON*(abs(B[j, j]) + abs(B[j+1, j+1])):
                B[j, j+1] = 0.0
        
        # Divide B into three block matrices
        # Determine smallest p and largest q for the block sizes
        B_33 = B[rows-q:, cols-q:]
        q_i = q
        #print(f"B_33 init (i={iteration_count+1}) = {B_33}")
        if abs(np.sum(B_33)) > EPSILON:
            for j in range(1, cols-1):
                B_33_temp = B[rows-q_i-j:, cols-q_i-j:]
                diff = B_33_temp-get_diagonal_copy(B_33_temp)
                if abs(np.sum(diff)) > EPSILON:
                    break
                B_33 = B_33_temp.copy()
                q += 1
        else:
            for j in range(1, cols-1):
                B_33_temp = B[rows-q_i-j:, cols-q_i-j:]
                diff = B_33_temp-get_diagonal_copy(B_33_temp)
                if abs(np.sum(diff)) > EPSILON:
                    break
                B_33 = B_33_temp.copy()
                q += 1

        B_22 = B[rows-q-1:rows-q, cols-1-q:cols-q]
        if abs(np.sum(B_22)) > EPSILON:
            p = cols-q-1
            for j in range(1, cols-q):
                B_22_temp = B[rows-1-q-j:rows-q, cols-1-q-j:cols-q]
                diff = B_22_temp-get_bidiagonal_copy(B_22_temp)
                p -= 1
                if abs(np.sum(diff)) > EPSILON:
                    break
                B_22 = B_22_temp.copy()
        else:
            for j in range(1, cols-q):
                B_22_temp = B[rows-1-q-j:rows-q, cols-1-q-j:cols-q]
                diff = B_22_temp-get_bidiagonal_copy(B_22_temp)
                if abs(np.sum(diff)) > EPSILON:
                    break
                B_22 = B_22_temp.copy()

        #B_11 = B[:p, :p]
        
        #print(f"B_22_{iteration_count+1} = {B_22}")
        #print(f"B_33_{iteration_count+1} = {B_33}")
        #print(f"q_{iteration_count+1} = {q}")
          
        if q < cols:
            # Zero out superdiagonal entry if diagonal entry is ~zero
            # via Givens rotation
            for j in range(p+1, cols-q):
                if abs(B[j, j]) < EPSILON:
                    G = givens_rotation(B.T, j+1, j)
                    B = B @ G.T
                    #print(f"B = {B}")
            U, B, V = golub_kahan_step(U, B, V, B.shape[1], p, q) 
        else:
            Σ = get_diagonal_copy(B)
            break
        
        if iteration_count == MAX_ITERATION_COUNT-1:
            print(f"A = {A}")
            raise RuntimeError(f"Algorithm was not able to converge to complete SVD for {MAX_ITERATION_COUNT} iterations!")
           
        #print(f"B_{iteration_count+1} = {B}")     
        iteration_count += 1
        #zero_out_small_numbers(B)
        
    #print(f"Iterations = {iteration_count}")
        
    return U, Σ, V.T
    
def verify_svd(A, U, Σ, Vt, suppress_success_flag=False):
    
    rows, cols = A.shape
    
    A_hat = U @ Σ @ Vt
    U_np, Σ_np, Vt_np = np.linalg.svd(A)
    np.around(A_hat, 2, A_hat)
        
    if not np.allclose(A_hat-A, np.zeros((rows, cols), dtype="float"), atol=1e-2):
        print(f"A = {A}")
        print(f"U = {U}")
        print(f"U_np = {U_np}")
        print(f"Σ = {Σ}")
        print(f"Σ_np = {Σ_np}")
        print(f"Vt = {Vt}")
        print(f"Vt_np = {Vt_np}")
        print(f"A_hat = {A_hat}")
        raise RuntimeError(f"One or more factors of the decomposition are incorrect!")
    
    if not suppress_success_flag:
        print(f"Computed factors from SVD are correct!")
        
def main():
    
    A = np.array([[  0. ,  0., -10. , -9.],
 [  7. , -7. , -3. ,  5.],
 [  9. ,-10.  ,-6.  , 5.],
 [  7.,   6. ,  9. ,  8.]]).astype("float")
    
    U, Σ, Vt = do_svd_via_golub_reinsch(A)
    
    verify_svd(A, U, Σ, Vt)
    
if __name__ == "__main__":
    main()