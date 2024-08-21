using SuiteSparseMatrixCollection
using MatrixMarket
using Plots

include("../src/LSRN.jl")

ssmc = ssmc_db()  
tol = 1e-10
# Extraire les matrices de la SuiteSparse Matrix Collection
landmark = ssmc[ssmc.name .== "landmark", :]
rail4284 = ssmc[ssmc.name .== "rail4284", :]

# Extraire les path des matrices
path = fetch_ssmc(landmark, format="MM")
path2 = fetch_ssmc(rail4284, format="MM")

# Charger les matrices avec HarwellRutherfordBoeing
landmark = MatrixMarket.mmread(joinpath(path[1], "$(landmark.name[1]).mtx"))
rail4284 = MatrixMarket.mmread(joinpath(path2[1], "$(rail4284.name[1]).mtx"))


# Définir les vecteurs b = A_i * rand, i = 1, 2
x_true_landmark = rand(size(landmark, 2))
x_true_rail4284 = rand(size(rail4284, 2))
b_landmark = landmark * x_true_landmark
b_rail4284 = rail4284 * x_true_rail4284

# Résoudre les problèmes de moindres carrés avec LSRN_l et LSRN_r

x_landmark, res_landmark = LSRN_l(landmark, b_landmark, γ = 1.5)
x_rail4284, res_rail4284 = LSRN_r(rail4284, b_rail4284, γ = 1.5)

x_landmark_lsqr, res_landmark_lsqr = LSRN_l(landmark, b_landmark, γ = 1.5, subsolver = :LSQR)
x_rail4284_lsqr, res_rail4284_lsqr = LSRN_r(rail4284, b_rail4284, γ = 1.5, subsolver = :LSQR)

# Use lsqr to solve the problem

x_landmark_lsqr1, stats_landmark_lsqr1 = lsqr(landmark, b_landmark, btol = tol, etol = tol, axtol = tol, history = true)
res_landmark_lsqr1 = stats_landmark_lsqr1.residuals
x_rail4284_lsqr1, stats_rail4284_lsqr1 = lsqr(rail4284, b_rail4284, btol = tol, etol = tol, axtol = tol, history = true)
res_rail4284_lsqr1 = stats_rail4284_lsqr1.residuals

# Use lsmr to solve the problem

x_landmark_lsmr, stats_landmark_lsmr = lsmr(landmark, b_landmark, btol = tol, etol = tol, axtol = tol, history = true)
res_landmark_lsmr = stats_landmark_lsmr.residuals
x_rail4284_lsmr, stats_rail4284_lsmr = lsmr(rail4284, b_rail4284, btol = tol, etol = tol, axtol = tol, history = true)
res_rail4284_lsmr = stats_rail4284_lsmr.residuals

# Afficher les résultats

println("#### Stats-lsrn-cs ####")
println("Résidus pour landmark lsrn cs: ", norm(landmark * x_landmark - b_landmark))
println("Norme de solution ", norm(x_landmark))
println("Résidus pour rail4284 lsrn cs: ", norm(rail4284 * x_rail4284 - b_rail4284))
println("Norme de solution ", norm(x_rail4284))

println("#### Stats-lsrn-lsqr ####")
println("Résidus pour landmark lsrn lsqr: ", norm(landmark * x_landmark_lsqr - b_landmark))
println("Norme de solution ", norm(x_landmark_lsqr))
println("Résidus pour rail4284 lsrn lsqr: ", norm(rail4284 * x_rail4284_lsqr - b_rail4284))
println("Norme de solution ", norm(x_rail4284_lsqr))

println("#### Stats-lsqr ####")
println("Résidus pour landmark lsqr1: ", norm(landmark * x_landmark_lsqr1 - b_landmark))
println("Norme de solution ", norm(x_landmark_lsqr1))
println("Résidus pour rail4284 lsqr1: ", norm(rail4284 * x_rail4284_lsqr1 - b_rail4284))
println("Norme de solution ", norm(x_rail4284_lsqr1))

println("#### Stats-lsmr ####")
println("Résidus pour landmark lsmr: ", norm(landmark * x_landmark_lsmr - b_landmark))
println("Norme de solution ", norm(x_landmark_lsmr))
println("Résidus pour rail4284 lsmr: ", norm(rail4284 * x_rail4284_lsmr - b_rail4284))
println("Norme de solution ", norm(x_rail4284_lsmr))

p = Plots.plot(1:length(res_landmark), log.(res_landmark), xlabel = "iter", ylabel = "residus", label = "LSRN-CS", legend = true)
plot!(p, 1:length(res_landmark_lsqr), log.(res_landmark_lsqr), label = "LSRN-LSQR", legend = true)
plot!(p, 1:length(res_landmark_lsqr1), log.(res_landmark_lsqr1), label = "LSQR", legend = true)
plot!(p, 1:length(res_landmark_lsmr), log.(res_landmark_lsmr), label = "LSMR", legend = true)

savefig(p, "res-landmark-cs.pdf") 

p2 = Plots.plot(1:length(res_rail4284), log.(res_rail4284), xlabel = "iter", ylabel = "residus", label = "LSRN-CS", legend = true)
plot!(p2, 1:length(res_rail4284_lsqr), log.(res_rail4284_lsqr), label = "LSRN-LSQR", legend = true)
plot!(p2, 1:length(res_rail4284_lsqr1), log.(res_rail4284_lsqr1), label = "LSQR", legend = true)
plot!(p2, 1:length(res_rail4284_lsmr), log.(res_rail4284_lsmr), label = "LSMR", legend = true)

savefig(p2, "res-rail4284-cs.pdf")

delete_all_ssmc()    