using SuiteSparseMatrixCollection
using MatrixMarket
using Plots

include("../src/LSRN.jl")

ssmc = ssmc_db()  

# Extraire les matrices de la SuiteSparse Matrix Collection
landmark = ssmc[ssmc.name .== "landmark", :]
rail4284 = ssmc[ssmc.name .== "rail4284", :]

# Extraire les path des matrices
path = fetch_ssmc(landmark, format="MM")
path2 = fetch_ssmc(rail4284, format="MM")

# Charger les matrices avec HarwellRutherfordBoeing
landmark = MatrixMarket.mmread(joinpath(path[1], "$(landmark.name[1]).mtx"))
rail4284 = MatrixMarket.mmread(joinpath(path2[1], "$(rail4284.name[1]).mtx"))

delete_all_ssmc()    

# Définir les vecteurs b = A_i * e, i = 1, 2
x_true_landmark = rand(size(landmark, 2))
x_true_rail4284 = rand(size(rail4284, 2))
b_landmark = landmark * x_true_landmark
b_rail4284 = rail4284 * x_true_rail4284

# Résoudre les problèmes de moindres carrés avec LSRN_l et LSRN_r

x_landmark, res_landmark = LSRN_l(landmark, b_landmark, γ = 1.5)
x_rail4284, res_rail4284 = LSRN_r(rail4284, b_rail4284, γ = 1.5)

x_landmark_lsqr, res_landmark_lsqr = LSRN_l(landmark, b_landmark, γ = 1.5, subsolver = :LSQR)
x_rail4284_lsqr, res_rail4284_lsqr = LSRN_r(rail4284, b_rail4284, γ = 1.5, subsolver = :LSQR)
# Afficher les résultats


println("Résidus pour landmark: ", norm(landmark * x_landmark - b_landmark))
println("Erreur relative pour landmark: ", norm(x_true_landmark - x_landmark) / norm(x_true_landmark))

p = Plots.plot(1:length(res_landmark), log.(res_landmark), xlabel = "iter", ylabel = "residus", label = "LSRN-CS", legend = true)
plot!(p, 1:length(res_landmark_lsqr), log.(res_landmark_lsqr), label = "LSRN-LSQR", legend = true)

savefig(p, "res-landmark-cs.pdf") 

println("Résidus pour rail4284: ", norm(rail4284 * x_rail4284 - b_rail4284))
println("Erreur relative pour rail4284: ", norm(x_true_rail4284 - x_rail4284) / norm(x_true_rail4284))

p2 = Plots.plot(1:length(res_rail4284), log.(res_rail4284), xlabel = "iter", ylabel = "residus", label = "LSRN-CS", legend = true)
plot!(p2, 1:length(res_rail4284_lsqr), log.(res_rail4284_lsqr), label = "LSRN-LSQR", legend = true)

savefig(p2, "res-rail4284-cs.pdf")
