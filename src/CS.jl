"""
    CS(A, b, σ_U, σ_L, ϵ, dim)
Solve the preconditioned least squares problem min_x ||Ax - b||_2 with the Chebyshev method, where A could be a dense matrix, a sparse matrix or a linear operator.
"""

function CS(op, b::AbstractVector{R}, σ_U::R, σ_L::R, ϵ::R, dim::Int) where {R <: Real}

    # Définition des constantes et vecteurs initiaux nécessaires pour l'algorithme de Chebyshev:

    d = (σ_U^2 + σ_L^2)/2
    c = (σ_U^2 - σ_L^2)/2

    y = zeros(R, dim) 
    v = zeros(R, dim)
    r = copy(b)

    α = R(0)
    β = R(0)

    # Nombre d'itérations max:
    iter = ceil(Int, (log(ϵ)-log(2)) / (log((σ_U-σ_L) / (σ_U+σ_L))) )
    L = R[]
    push!(L, norm(r))

    # Début de la boucle itérative de Chebyshev:
    for k in 0:iter
        if k == 0
            α = 1/d
            β = R(0)

        elseif k == 1
            α = d - (c^2) / 2*d
            β = 1/2 * (c/d)^2

        else
            α = 1/(d-α * (c^2)/4)
            β = (α * c/2)^2
        end

        mul!(v, op', r, R(1), β)
        @. y += α * v
        mul!(r, op, v, -α, R(1))
        push!(L, norm(r))

    end
    return y, L

end
