function CS_l(A, b, σ_U, σ_L, ϵ, N, dim)

    # Définition des constantes et vecteurs initiaux nécessaires pour l'algorithme de Chebyshev:

    d = (σ_U^2 + σ_L^2)/2
    c = (σ_U^2 - σ_L^2)/2

    α = 0
    β = 0

    y = zeros(dim) 
    v = zeros(dim)
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

        v = β * v + N' * A' * r
        y = y + α * v
        r = r - α * A * N * v

    end
    return y

end

function CS_r(A, b, σ_U, σ_L, ϵ, M, dim)

    # Définition des constantes et vecteurs initiaux nécessaires pour l'algorithme de Chebyshev:

    d = (σ_U^2 + σ_L^2)/2
    c = (σ_U^2 - σ_L^2)/2

    α = 0
    β = 0

    y = zeros(dim) 
    v = zeros(dim)
    r = M' * b

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

        v = β * v + A' * M * r
        y = y + α * v
        r = r - α * M' * A * v

    end
    return y

end
