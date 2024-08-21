using Plots

include("../src/LSRN.jl")

Random.seed!(1234)

# Test LSRN_l with big and sparse matrix

m = 100000
n = 1000
x = rand(n)
A = sprand(m, n, 0.001)
b = A * x
println("norme de x " , norm(x))

x̂_CS_l, L_CS_l, t_CS_l = LSRN_l(A,b)

println("Résidus CS pour A*x̂ - b: ", norm(A*x̂_CS_l - b))
println("Erreur relative CS pour x: ", norm(x - x̂_CS_l)/norm(x))
println("Temps CS: ", t_CS_l)
println("norme de x̂_CS_l " , norm(x̂_CS_l))

x̂_LSQR_l, L_LSQR_l, t_LSQR_l = LSRN_l(A,b, subsolver = :LSQR)

println("Résidus LSQR pour A*x̂ - b: ", norm(A*x̂_LSQR_l - b))
println("Erreur relative LSQR pour x: ", norm(x - x̂_LSQR_l)/norm(x))
println("Temps LSQR: ", t_LSQR_l)
println("norme de x̂_LSQR_l " , norm(x̂_LSQR_l))


p = Plots.plot(1:length(L_CS_l), log.(L_CS_l), xlabel = "iter", ylabel = "residus", label = "CS")
plot!(p, 1:length(L_LSQR_l), log.(L_LSQR_l), label = "LSQR")

savefig(p, "sparse-left-lsqr-cs.pdf")

# Test LSRN_l with big and sparse matrix

x = rand(m)
Ar = sprand(n, m, 0.001)
b = Ar * x
println("norme de x " , norm(x))

x̂_CS_r, L_CS_r, t_CS_r = LSRN_r(Ar,b)

println("Résidus CS pour Ar*x̂ - b: ", norm(Ar*x̂_CS_r - b))
println("Erreur relative CS pour x: ", norm(x - x̂_CS_r)/norm(x))
println("Temps CS: ", t_CS_r)
println("norme de x̂_CS_r " , norm(x̂_CS_r))

x̂_LSQR_r, L_LSQR_r, t_LSQR_r = LSRN_r(Ar,b, subsolver = :LSQR)

println("Résidus LSQR pour Ar*x̂ - b: ", norm(Ar*x̂_LSQR_r - b))
println("Erreur relative LSQR pour x: ", norm(x - x̂_LSQR_r)/norm(x))
println("Temps LSQR: ", t_LSQR_r)
println("norme de x̂_LSQR_r " , norm(x̂_LSQR_r))

p2 = Plots.plot(1:length(L_CS), log.(L_CS), xlabel = "iter", ylabel = "residus", label = "CS")
plot!(p2, 1:length(L_LSQR_r), log.(L_LSQR_r), label = "LSQR")

savefig(p2, "sparse-right-lsqr-cs.pdf")
