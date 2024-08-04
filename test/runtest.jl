# Test
using Test

include("../src/LSRN.jl")
include("../src/CS.jl")

m = 100000
n = 50
A = rand(m,n)

x = ones(n)
b = A*x

x̂ = LSRN_l(A,b)


@test norm(A*x̂ - b) < 1e-2
@test norm(x - x̂)/norm(x) < 1e-4

B = sprand(m,n,0.1)
b = B*x

x̂ = LSRN_l_sparse(B,b)

@test norm(B*x̂ - b) < 1e-2
@test norm(x - x̂)/norm(x) < 1e-4

# Test LSRN_r

# Ar = rand(n,m)

# x = ones(m)
# b = Ar*x

# x̂ = LSRN_r(Ar,b)

# @test norm(Ar*x̂ - b) < 1e-2
# @test norm(x - x̂)/norm(x) < 1e-2

# Br = sprand(n,m,0.1)

# b = Br*x

# x̂ = LSRN_r_sparse(Br,b)

# @test norm(Br*x̂ - b) < 1e-2
# @test norm(x - x̂)/norm(x) < 1e-2


# Le test n'est plus bon
# A = rand(5, 5)
# b = rand(5)
# ε = 1e-6

# x = CS(A, b, ε)
# @test norm(A*x-b) < 1e-3