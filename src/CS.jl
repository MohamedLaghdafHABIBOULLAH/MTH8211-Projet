using LinearAlgebra
using Krylov
using Random
using SparseArrays
using PROPACK


function CS(F, b, σ_U, σ_L, tol)
    m,n = size(F)
    ϵ = tol
    # Définition des constantes et vecteurs initiaux nécessaires pour l'algorithme de Chebyshev:

    d = (σ_U^2 + σ_L^2)/2
    c = (σ_U^2 - σ_L^2)/2

    α = 0
    β = 0

    y = zeros(n) 
    v = zeros(n)
    r = b

    # Nombre d'itérations max:
    iter = ceil(Int, (log(ϵ)-log(2)) / (log((σ_U-σ_L) / (σ_U+σ_L))) )

    # Début de la boucle itérative de Chebyshev:
    for k in 0:iter
        if k == 0
            α = 1/d
            β = 0

        elseif k ==1
            α = d - (c^2) / 2*d
            β = 1/2 * (c/d)^2

        else
            α = 1/(d-α * (c^2)/4)
            β = (α * c/2)^2
        end

        v = β * v + F' * r
        y = y + α * v
        r = r - α * F * v

    end
    println("Méthode CS")
    return y

end