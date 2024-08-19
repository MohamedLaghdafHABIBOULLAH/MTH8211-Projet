# MTH8211-Projet

# Least Squares Random  Normalization (LSRN)

## Introduction

LSRN (Least Squares with Random Normal projection) is a robust, efficient, and scalable algorithm designed to solve large-scale least squares problems. It leverages randomized projections and iterative refinement to achieve high accuracy while maintaining computational efficiency. This method is particularly useful for solving over-determined systems where the number of equations exceeds the number of unknowns.

## Key Features

- **Scalability**: LSRN is highly scalable and can handle very large datasets, making it suitable for big data applications.
- **Accuracy**: The method provides accurate solutions, even for ill-conditioned matrices, due to its normalization step and iterative refinement.
- **Randomized Projections**: LSRN uses randomized projections to reduce the dimensionality of the problem, making the computation more manageable.
- **Iterative Refinement**: The algorithm includes an iterative refinement step to improve the accuracy of the solution, especially in the presence of numerical errors.

## How LSRN Works

### Problem Statement

Given a matrix $A \in \mathbb{R}^{m \times n}$ and a vector $\vec{b} \in \mathbb{R}^m $, the least squares problem is to find $\vec{x} \in \mathbb{R}^n$ that minimizes the following:

$\min_{\vec{x}} \vert A\vec{x}- \vec{b} \vert_2$

### Algorithm Overview

LSRN solves this problem using the following steps:

1. **Randomized Normalization**: Apply a random Gaussian matrix $G \in \mathbb{R}^{t \times m}$ to the matrix $A$ to form the matrix $G \times A$. This step normalizes the condition number of the problem, making it easier to solve.
   
2. **QR Decomposition**: Compute the QR decomposition of the matrix $G \times A$ to obtain orthogonal matrices $Q$ and upper triangular matrix $R$.
   
3. **Solve the Triangular System**: Solve the triangular system $R \vec{y} = Q^T \vec{b}$ to find the intermediate solution $\vec{y}$.

4. **Iterative Refinement**: Refine the solution by solving the original least squares problem iteratively, using the intermediate solution $\vec{y}$ as a starting point.

5. **Final Solution**: Compute the final solution $\vec{x} = R^{-1} \vec{y}$.
<!--
### Pseudocode

Input: A matrix $A^{m \times n}$, vector $\vec{b}$ of length m
Output: Vector $\vec{x}$ that minimizes $||A\vec{x} - \vec{b}|| $
## Algorithm 1: $\text{LSRN}_{left}$ $(A, \vec{b}, γ = 2, tol = 10^{-14})$ — case where $n ≪ m$

1. Choose an oversampling factor γ, with $\gamma > 1$, and set s = ⌈γn⌉.
2. Generate a random matrix from a normal distribution $G = \text{rand}(s, m)$.
3. Compute the SVD of the simplified matrix $\tilde{A} = GA$, denoted as $\tilde{U} \tilde{\Sigma} \tilde{V}$.
4. Set $N = \tilde{V} \tilde{\Sigma}^{-1}$.
5. Use an iterative sub-solver to solve the preconditioned problem $\min_y \vert AN\vec{y} - \vec{b} \vert_2$ and denote the solution as $\hat{y}$.
6. Return $\hat{x} = N\hat{y}$.

---

## Algorithm 2: LSRN_right $(A, \vec{b}, γ = 2, tol = 10^{-14})$ — case where m ≪ n

1. Choose an oversampling factor γ, with $\gamma > 1$, and set s = ⌈γm⌉.
2. Generate a random matrix from a normal distribution $G = \text{rand}((s, n))$.
3. Compute the SVD of the simplified matrix $\tilde{A} = AG$, denoted as $\tilde{U} \tilde{\Sigma} \tilde{V}$.
4. Set $M = \tilde{U} \tilde{\Sigma}^{-1}$.
5. Use an iterative sub-solver to solve the preconditioned problem $\min_x \| M^T Ax - M^T b \|_2$ and denote the solution as $\hat{x}$.
6. Return $\hat{x}$.

---

## Algorithm 3: Chebyshev semi-iterative (CS) method (Meng et al., 2014)

1. Given a matrix $A \in \mathbb{R}^{m \times n}$, let $r = \text{rank}(A)$, a vector $\vec{b} \in \mathbb{R}^m$, and a tolerance $\epsilon > 0$, choose $0 < \sigma_L \leq \sigma_U$ such that all nonzero singular values of $A$ are in $[\sigma_L, \sigma_U]$.
2. Set $d = (\sigma_U^2 + \sigma_L^2) / 2, c = (\sigma_U^2 - \sigma_L^2) / 2 , \vec{x}_0 = 0, \vec{v}_0 = 0,$ and $\vec{r}_0 = \vec{b}$.
3. For $k = 0, 1, \dots, \lceil (\log \epsilon - \log 2) / \log(\sigma_U - \sigma_L) / (\sigma_U + \sigma_L) \rceil$, do:
   - Set:
     - $\alpha \leftarrow 1 / d$ if $k = 0$,
     - $\alpha \leftarrow d - c^2 / (2d)$ if $k = 1$,
     - $\alpha \leftarrow 1 / (d - \alpha c^2 / 4)$ otherwise.
   - Set:
     - $\beta \leftarrow 0$ if $k = 0$,
     - $\beta \leftarrow (c/d)^2 / 2$ if $k = 1$,
     - $\beta \leftarrow (\alpha c / 2)^2$ otherwise.
4. Update:
   - $\vec{v}_k \leftarrow \beta \vec{v}_k + A^T \vec{r}_k$,
   - $\vec{x}_k \leftarrow \vec{x}_k + \alpha \vec{v}_k$,
   - $\vec{r}_k \leftarrow \vec{r}_k - \alpha A \vec{v}_k$.
5. End loop.
-->

## Reference

