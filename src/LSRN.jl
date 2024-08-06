using LinearAlgebra
using Krylov
using Random
using SparseArrays

"""
LSRN_l(A, b; γ = 2, tol = 1e-14)
    LSRN resolves the minimum norm solution of the equation min ||x||_2 s.t. x in argmin ||Ax-b||_2, where A is a tall matrix.
"""

"""
LSRN_r(A, b; γ = 2, tol = 1e-14)
    LSRN resolves the minimum norm solution of the equation min ||x||_2 s.t. x in argmin ||Ax-b||_2, where A is a wide matrix.
"""

include("CS.jl")
include("utils.jl")

function LSRN_l(A , b; γ = 2, tol = 1e-14, subsolver = :CS)
    m,n = size(A)
    @assert m > n
    
    # set s = ⌈γn⌉.
    s = ceil(Int, γ*n)

    # Compute A1 = GA.
    A1 = Generate_GA(A, m, n, s)
    
    # Compute SVD of A1.
    _, Σ_1, V_1 = svd(A1)

    # extract the rank of A1, length of Σ_1
    r = length(Σ_1)

    # Let N=V _1 Σ_1^−1.
    P_1 = Diagonal(1 ./ Σ_1)
    N = V_1 * P_1

    # Obtaining highest and lowest singular values of AN:
    # TH 4.3: For any ̃α ∈ [0,1-√(r/s)] (in this case r=n):
    a = 1e-8 # Le choix a = 0. proposé par l'article fonctionne qu'en grande dimension 10^6 * 10^3

    σ_U = 1 / ((1-a)*sqrt(s)-sqrt(r))
    σ_L = 1 / ((1+a)*sqrt(s)+sqrt(r))

    # Compute min-length solution: 
    if subsolver == :CS
        y, L = CS_l(A, b, σ_U, σ_L, tol, N, n)
    elseif subsolver == :LSQR
        y, stats = lsqr(A*N, b, axtol = tol, btol = tol, etol = tol,  history = true)
        L = stats.residuals
    else
        error("Unknown subsolver")
    end

    return N * y, L
end

function LSRN_r(A , b; γ = 2,  tol = 1e-14, subsolver = :CS)
    m,n = size(A)
    @assert m < n
    
    # set s = ⌈γm⌉.
    s = ceil(Int, γ*m)
    
    # Compute A1 = GA.
    A1 = Generate_AG(A, m, n, s)
    
    # Compute SVD of A1.
    U_1, Σ_1, _ = svd(A1)

    # extract the rank of A1, length of Σ_1
    r = length(Σ_1)
    
    # Let M =U _1 Σ_1^−1.
    P_1 = Diagonal(1 ./ Σ_1)
    M = U_1 * P_1

    # Obtaining highest and lowest singular values of AN:
    # TH 4.3: For any ̃α ∈ [0,1-√(r/s)] (in this case r=n):
    a = 1e-8 # Le choix a = 0. proposé par l'article fonctionne qu'en grande dimension 10^6 * 10^3

    σ_U = 1 / ((1-a)*sqrt(s)-sqrt(r))
    σ_L = 1 / ((1+a)*sqrt(s)+sqrt(r))

    # Compute min-length solution: 
    x, L = CS_r(A, b, σ_U, σ_L, tol, M, n)
    if subsolver == :CS
        x, L = CS_r(A, b, σ_U, σ_L, tol, M, n)
    elseif subsolver == :LSQR
        x, stats = lsqr(M' * A, M' * b, axtol = tol, btol = tol, etol = tol, history = true)
        L = stats.residuals
    else
        error("Unknown subsolver")
    end
    
    return x, L
end
