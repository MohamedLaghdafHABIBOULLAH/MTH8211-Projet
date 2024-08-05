using Test

include("../src/LSRN.jl")

Random.seed!(1234)

m = 100000
n = 50
A = rand(m,n)
x = ones(n)
b = A*x

x̂ = LSRN_l(A,b)

@test norm(A*x̂ - b) < 1e-8
@test norm(x - x̂)/norm(x) < 1e-10

# Test LSRN_l with big and sparse matrix

m = 1000000
n = 1000
x = rand(n)
A = sprand(m, n, 0.001)
b = A * x

x̂ = LSRN_l(A,b)

@test norm(A*x̂ - b) < 1e-8
@test norm(x - x̂)/norm(x) < 1e-10

# Test LSRN_r autre pull !

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
