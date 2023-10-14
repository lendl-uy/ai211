# QR Decomposition with Column Pivoting
## AI 211 Coding Challenge 3

This code submission by Jan Lendl Uy comprises two files:

- QR_Decomposition.py
- QR_Decomposition_Tests.py

In `QR_Decomposition.py`, the different methods involved in performing AP = QR factorization via the Gram-Schmidt process are found in this file. You may check this file to inspect how the algorithm was implemented.

On the other hand, `QR_Decomposition_Tests.py` houses the tests performed in verifying the correctness of the QR decomposition algorithm written by Lendl. RUN this file to check the results of the factorization for different tests which include:

- A is the zero matrix
- A is square and has full rank
- A is square and is rank-deficient
- Given an mxn A matrix, m > n
- Given an mxn A matrix, n > m
- Randomized testing of the QR algorithm