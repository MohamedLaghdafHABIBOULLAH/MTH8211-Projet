# MTH8211-Projet

# Least Squares Regression with Normalization (LSRN)

## Introduction

LSRN (Least Squares Regression with Normalization) is a robust, efficient, and scalable algorithm designed to solve large-scale least squares problems. It leverages randomized projections and iterative refinement to achieve high accuracy while maintaining computational efficiency. This method is particularly useful for solving over-determined systems where the number of equations exceeds the number of unknowns.

## Key Features

- **Scalability**: LSRN is highly scalable and can handle very large datasets, making it suitable for big data applications.
- **Accuracy**: The method provides accurate solutions, even for ill-conditioned matrices, due to its normalization step and iterative refinement.
- **Randomized Projections**: LSRN uses randomized projections to reduce the dimensionality of the problem, making the computation more manageable.
- **Iterative Refinement**: The algorithm includes an iterative refinement step to improve the accuracy of the solution, especially in the presence of numerical errors.

## How LSRN Works

### Problem Statement

Given a matrix $\( A \in \mathbb{R}^{m \times n} \)$ and a vector \( b \in \mathbb{R}^m \), the least squares problem is to find \( x \in \mathbb{R}^n \) that minimizes the following:

\[
\min_x \| Ax - b \|_2
\]

### Algorithm Overview

LSRN solves this problem using the following steps:

1. **Randomized Normalization**: Apply a random Gaussian matrix \( G \in \mathbb{R}^{t \times m} \) to the matrix \( A \) to form the matrix \( G \times A \). This step normalizes the condition number of the problem, making it easier to solve.
   
2. **QR Decomposition**: Compute the QR decomposition of the matrix \( G \times A \) to obtain orthogonal matrices \( Q \) and upper triangular matrix \( R \).
   
3. **Solve the Triangular System**: Solve the triangular system \( R \times y = Q^T \times b \) to find the intermediate solution \( y \).

4. **Iterative Refinement**: Refine the solution by solving the original least squares problem iteratively, using the intermediate solution \( y \) as a starting point.

5. **Final Solution**: Compute the final solution \( x = R^{-1} \times y \).

### Pseudocode

```text
Input: A matrix A of size m x n, vector b of length m
Output: Vector x that minimizes ||Ax - b||

1. Generate a random Gaussian matrix G of size t x m
2. Compute the matrix product B = G * A
3. Perform QR decomposition: [Q, R] = qr(B)
4. Solve the system R * y = Q^T * b
5. Perform iterative refinement to improve the solution
6. Compute the final solution x = R^{-1} * y
