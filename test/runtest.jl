using Test

include("../src/LSRN.jl")

Random.seed!(1234)

# Test LSRN_l

m = 100000
n = 50
A = rand(m,n)
x = ones(n)
b = A*x

x̂, _ = LSRN_l(A,b)

@test norm(A*x̂ - b) < 1e-8
@test norm(x - x̂)/norm(x) < 1e-10

# Test LSRN_r 

m = 100000
n = 50

Ar = rand(Float32, n,m)

x = ones(Float32, m)
b = Ar*x

x̂, _ = LSRN_r(Ar,b)

@test norm(Ar*x̂ - b) < 1e-6
@test norm(x - x̂)/norm(x) < 1e-1
@test typeof(x̂) == Vector{Float32}
