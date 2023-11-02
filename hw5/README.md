# Computing the Eigenvalues and Eigenvectors of a Matrix via QR Algorithm
## AI 211 Coding Challenge 5

This code submission by Jan Lendl Uy comprises two files:

- Get_Eigenvalues.py
- Get_Eigenvalues_Tests.py

In `Get_Eigenvalues.py`, the algorithm for computing the eigenvalues and eigenvectors (if the matrix is symmetric) via the QR algorithm is found in this program. You may check this file to inspect the implementation of the algorithm.

On the other hand, `Get_Eigenvalues_Tests.py` contains the tests performed in verifying the correctness of the algorithm for computing the eigenvalues and eigenvectors. RUN this file to check the results for different tests which include:

- A is the {[0, 1], [1, 0]} matrix where standard QR algorithm fails
- A is a symmetric 3x3 matrix
- Randomized testing for a symmetric matrix with random mxm size and random entries