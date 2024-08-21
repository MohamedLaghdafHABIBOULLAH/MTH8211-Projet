using LinearAlgebra
using Krylov
using Random
using LinearOperators

"""
LSRN_l(A, b; γ = 2, tol = 1e-10)
    LSRN resolves the minimum norm solution of the equation min ||x||_2 s.t. x in argmin ||Ax-b||_2, where A is a tall matrix.
"""

"""
LSRN_r(A, b; γ = 2, tol = 1e-10)
    LSRN resolves the minimum norm solution of the equation min ||x||_2 s.t. x in argmin ||Ax-b||_2, where A is a wide matrix.
"""

include("CS.jl")
include("utils.jl")

function LSRN_l(A::AbstractArray{R}, b::AbstractVector{R}; γ::Float64 = 2., tol::R = R(1e-10), subsolver = :CS) where {R <: Real}
    m,n = size(A)
    @assert m >= n
    
    # set s = ⌈γn⌉.
    s = ceil(Int, γ*n)

    # Compute A1 = GA.
    A1 = Generate_GA(A, m, n, s, R)
    
    # Compute SVD of A1.
    _, Σ_1, V_1 = svd(A1)

    # extract the rank of A1, length of Σ_1
    r = length(Σ_1)

    # Let N=V _1 Σ_1^−1.
    P_1 = Diagonal(1 ./ Σ_1)
    N = V_1 * P_1

    # Obtaining highest and lowest singular values of AN:
    # TH 4.3: For any ̃α ∈ [0,1-√(r/s)] (in this case r=n):
    a = R(1e-8) # Le choix a = 0. proposé par l'article fonctionne qu'en grande dimension 10^6 * 10^3

    σ_U = 1 / ((1-a)*sqrt(R(s))-sqrt(R(r)))
    σ_L = 1 / ((1+a)*sqrt(R(s))+sqrt(R(r)))

    op = LinearOperator( R, m, r, false, false, 
                        (res, v) -> mul!(res, A, N * v),
                        (res, w) -> mul!(res, N', A' * w))

    # Compute min-length solution: 
    if subsolver == :CS
        t = @elapsed y, L = CS(op, b, σ_U, σ_L, tol, n)
    elseif subsolver == :LSQR     
        t = @elapsed y, stats = lsqr(op, b, atol = tol, axtol = tol, btol = tol, etol = tol, history = true)
        L = stats.residuals
    else
        error("Unknown subsolver")
    end

    return N * y, L, t
end

function LSRN_r(A::AbstractArray{R}, b::AbstractVector{R}; γ::Float64 = 2., tol::R = R(1e-10), subsolver = :CS) where {R <: Real}
    m,n = size(A)
    @assert m < n
    
    # set s = ⌈γm⌉.
    s = ceil(Int, γ*m)
    
    # Compute A1 = AG.
    A1 = Generate_AG(A, m, n, s, R)
    
    # Compute SVD of A1.
    U_1, Σ_1, _ = svd(A1)

    # extract the rank of A1, length of Σ_1
    r = length(Σ_1)
    
    # Let M =U _1 Σ_1^−1.
    P_1 = Diagonal(1 ./ Σ_1)
    M = U_1 * P_1

    # Obtaining highest and lowest singular values of AN:
    # TH 4.3: For any ̃α ∈ [0,1-√(r/s)] (in this case r=n):
    a = R(1e-8) # Le choix a = 0. proposé par l'article fonctionne qu'en grande dimension 10^6 * 10^3

    σ_U = 1 / ((1-a)*sqrt(R(s))-sqrt(R(r)))
    σ_L = 1 / ((1+a)*sqrt(R(s))+sqrt(R(r)))

    # Define the linear operator
    op = LinearOperator( R, r, n, false, false, 
                        (res, v) -> mul!(res, M', A * v),
                        (res, w) -> mul!(res, A', M * w))

    # Compute min-length solution: 
    if subsolver == :CS
        t = @elapsed x, L = CS(op, M' * b, σ_U, σ_L, tol, n)
    elseif subsolver == :LSQR
        t = @elapsed x, stats = lsqr(op, M' * b, atol = tol, axtol = tol, btol = tol, etol = tol, history = true)
        L = stats.residuals
    else
        error("Unknown subsolver")
    end
    
    return x, L, t
end
