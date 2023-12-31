# Performing Bidiagonalization and Tridiagonalization for SVD
## AI 211 Coding Challenge 6

This code submission by Jan Lendl Uy comprises two files:

- Do_Golub_Kahan_Step.py
- Do_Golub_Kahan_Step_Tests.py

In `Do_SVD.py`, it contains the implementations for performing bidiagonalization via the Golub-Kahan algorithm and tridiagonalization given a bidiagonal matrix. You may check this file to inspect the implementation of the two routines.

On the other hand, `Do_SVD_Tests.py` contains the tests performed in verifying the correctness of the bidiagonal and tridiagonal matrices computed by the codes written by the author. RUN this file to check the results for different tests which include:

- Bidiagonalize and tridiagonalize a 3x3 matrix
- Randomized testing of bidiagonalization and tridiagonalization for a matrix with random size and entries