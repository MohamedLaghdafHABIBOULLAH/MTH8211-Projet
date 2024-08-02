using LinearAlgebra
using Krylov
using Random
using SparseArrays
using PROPACK

"""
LSRN_l(A, b; γ = 3, maxiter = 100, tol = 1e-8)
    LSRN resolves the minimum norm solution of the equation min ||x||_2 s.t. x in argmin ||Ax-b||_2
There are several variants of  LSRN_l, LSRN_r, LSRN_l_sparse, LSRN_r_sparse.
"""


function LSRN_l(A , b; γ = 3, tol = 1e-8)
    m,n = size(A)
    @assert m > n
    
    # set s = ⌈γn⌉.
    s = ceil(Int, γ*n)

    # Generate G = randn(s,m)
    G = randn(s,m)

    # Compute A1 = GA.
    A1 = G * A

    # Compute SVD of A1.
    _, Σ_1, V_1 = svd(A1)

    # Let N=V _1 Σ_1^−1.
    P_1 = Diagonal(1 ./ Σ_1)
    N = V_1*P_1

    # Define a regularization parameter.
    # λ = 1.0e-3

    y, _ = lsqr(A*N, b, atol = tol, btol=tol)
    
    return N*y
end

function LSRN_l_sparse(A , b; γ = 3,  tol = 1e-8)
    m,n = size(A)
    @assert m > n
    
    # set s = ⌈γn⌉.
    s = ceil(Int, γ*n)

    # Generate G = randn(s,m).
    G = randn(s,m)

    # Compute A1 = GA.
    A1 = G * A
    
    # Compute SVD of A1.
    _, Σ_1, V_1 = tsvd(A1, k = n)
    
    # Let N=V _1 Σ_1^−1.
    P_1 = Diagonal(1 ./ Σ_1)
    N = V_1*P_1
    
    # Define a regularization parameter.
    # λ = 1.0e-3

    y, _ = lsqr(A*N, b, atol = tol, btol=tol)
    
    return N*y
end


function LSRN_r(A , b; γ = 5,  tol = 1e-8)
    m,n = size(A)
    @assert m < n
    
    # set s = ⌈γm⌉.
    s = ceil(Int, γ*m)
    
    # Generate G = randn(s,m)
    G = randn(n,s)
    
    # Compute A1 = GA.
    A1 = A * G
    
    # Compute SVD of A1.
    U_1, Σ_1, _ = svd(A1)
    
    # Let M =U _1 Σ_1^−1.
    P_1 = Diagonal(1 ./ Σ_1)
    M = U_1 * P_1

    # Define a regularization parameter.
    # λ = 1.0e-3

    x, _ = lsqr(M' * A, M' * b, atol = tol, btol=tol)
    
    return x
end

function LSRN_r_sparse(A , b; γ = 5, tol = 1e-8)
    m,n = size(A)
    @assert m < n
    
    # set s = ⌈γm⌉.
    s = ceil(Int, γ*m)
    
    # Generate G = randn(s,m)
    G = randn(n,s)
    
    # Compute A1 = GA.
    A1 = A * G
    
    # Compute SVD of A1.
    U_1, Σ_1, _ = tsvd(A1, k = m)
    
    # Let M=U _1 Σ_1^−1.
    P_1 = Diagonal(1 ./ Σ_1)
    M = U_1 * P_1

    # Define a regularization parameter.
    # λ = 1.0e-3

    x, _ = lsqr(M' * A, M' * b, atol = tol, btol=tol)
    
    return x
end