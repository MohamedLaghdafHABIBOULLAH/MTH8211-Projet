using LinearAlgebra
using Krylov
using Random
using SparseArrays
using PROPACK


#=
Choisir σ_U et σ_L tq. 0 < σ_L ≤ σ_U pour que σ_i(A) ∈ [σ_L, σ_U] ∀ σ_i ≠ 0
=#

function CS(A, b, ϵ)
    Σ = svd(A).S

    # Trouve la plus grande v.s. de A et une deuxième valeur inférieure à celle-ci:
    σ_U = Σ[1]   
    σ_L = rand(Σ)

    # Définition des constantes et vecteurs initiaux nécessaires pour l'algorithme de Chebyshev:
    m,n = size(A)
    d = (σ_U^2 + σ_L^2)/2
    c = (σ_U^2 - σ_L^2)/2

    α = 0
    β = 0

    x = zeros(n) 
    v = zeros(n)
    r = b
    # Nombre d'itérations max:
    iter = ceil(Int, (log(ϵ)-log(2)) / (log((σ_U-σ_L) / (σ_U+σ_L))) )

    # Début de la boucle itérative de Chebyshev:
    for k in 0:iter
        if k == 0
            β = 0
            α = 1/d

        elseif k ==1
            β = 1/2 * (c/d)^2
            α = d - (c^2) / 2*d

        else
            β = (α * c/2)^2
            α = 1/((d-α * c^2)/4)
        end

        v = β * v + A' * r
        x = x + α * v
        r = r - α * A * v
    end

    return x

end