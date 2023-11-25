# Performing SVD via Golub-Kahan and Golub-Reinsch
## AI 211 Coding Challenge 7

This code submission by Jan Lendl Uy comprises two files:

- Do_SVD.py
- Do_SVD_Tests.py

In `Do_SVD.py`, it contains the implementations for performing singular value decomposition using the Golub-Kahan algorithm and Golub-Reinsch algorithm. You may check this file to inspect the implementation of the two routines.

On the other hand, `Do_SVD_Tests.py` contains the tests performed in verifying the correctness of the decomposed matrix. RUN this file to check the results for different tests which include:

- 4x4 dense matrix (all elements are nonzero)
- 4x4 bidiagonal matrix
- Randomized testing of bidiagonalization and tridiagonalization for a matrix with random size and entries