using SuiteSparseMatrixCollection
using MatrixMarket
using Plots

include("../src/LSRN.jl")

ssmc = ssmc_db()  
tol = 1e-10
# Extraire les matrices de la SuiteSparse Matrix Collection
landmark = ssmc[ssmc.name .== "landmark", :]
rail4284 = ssmc[ssmc.name .== "rail4284", :]
rail582 = ssmc[ssmc.name .== "rail582", :]
specular = ssmc[ssmc.name .== "specular", :]
stat96v1 = ssmc[ssmc.name .== "stat96v1", :]

# Extraire les path des matrices
path = fetch_ssmc(landmark, format="MM")
path2 = fetch_ssmc(rail4284, format="MM")
path3 = fetch_ssmc(rail582, format="MM")
path4 = fetch_ssmc(specular, format="MM")
path5 = fetch_ssmc(stat96v1, format="MM")

# Charger les matrices avec HarwellRutherfordBoeing
landmark = MatrixMarket.mmread(joinpath(path[1], "$(landmark.name[1]).mtx"))
rail4284 = MatrixMarket.mmread(joinpath(path2[1], "$(rail4284.name[1]).mtx"))
rail582 = MatrixMarket.mmread(joinpath(path3[1], "$(rail582.name[1]).mtx"))
specular = MatrixMarket.mmread(joinpath(path4[1], "$(specular.name[1]).mtx"))
stat96v1 = MatrixMarket.mmread(joinpath(path5[1], "$(stat96v1.name[1]).mtx"))

delete_all_ssmc()    

function solve_lsrn_l(matrix, name)
    tol = 1e-10
    x_true = rand(size(matrix, 2))
    b = matrix * x_true
    x, res = LSRN_l(matrix, b, γ = 1.5)
    x_lsqr, res_lsqr = LSRN_l(matrix, b, γ = 1.5, subsolver = :LSQR)
    x_lsqr1, stats_lsqr1 = lsqr(matrix, b, btol = tol, etol = tol, axtol = tol, history = true, itmax = 10000)
    res_lsqr1 = stats_lsqr1.residuals
    x_lsmr, stats_lsmr = lsmr(matrix, b, btol = tol, etol = tol, axtol = tol, history = true, itmax = 10000)
    res_lsmr = stats_lsmr.residuals
    println("#### Stats-lsrn-cs ####")
    println("Résidus pour lsrn cs: ", norm(matrix * x - b))
    println("Norme de solution ", norm(x))
    println("Iterations: ", length(res))
    println("#### Stats-lsrn-lsqr ####")
    println("Résidus pour lsrn lsqr: ", norm(matrix * x_lsqr - b))
    println("Norme de solution ", norm(x_lsqr))
    println("Iterations: ", length(res_lsqr))
    println("#### Stats-lsqr ####")
    println("Résidus pour lsqr1: ", norm(matrix * x_lsqr1 - b))
    println("Norme de solution ", norm(x_lsqr1))
    println("Iterations: ", length(res_lsqr1))
    println("#### Stats-lsmr ####")
    println("Résidus pour lsmr: ", norm(matrix * x_lsmr - b))
    println("Norme de solution ", norm(x_lsmr))
    println("Iterations: ", length(res_lsmr))
    p = Plots.plot(1:length(res), log.(res), label = "LSRN-CS", legend = true, xlabel = "iter", ylabel = "residus")
    plot!(p, 1:length(res_lsqr), log.(res_lsqr), label = "LSRN-LSQR", legend = true)
    plot!(p, 1:length(res_lsqr1), log.(res_lsqr1), label = "LSQR", legend = true)
    plot!(p, 1:length(res_lsmr), log.(res_lsmr), label = "LSMR", legend = true)
    savefig(p, "res-"*name*".pdf")
end

# Pareil pour lsrn_r à la place de lsrn_l

function solve_lsrn_r(matrix, name)
    tol = 1e-10
    x_true = rand(size(matrix, 2))
    b = matrix * x_true
    x, res = LSRN_r(matrix, b, γ = 1.5)
    x_lsqr, res_lsqr = LSRN_r(matrix, b, γ = 1.5, subsolver = :LSQR)
    x_lsqr1, stats_lsqr1 = lsqr(matrix, b, btol = tol, etol = tol, axtol = tol, itmax = 10000, history = true)
    res_lsqr1 = stats_lsqr1.residuals
    x_lsmr, stats_lsmr = lsmr(matrix, b, btol = tol, etol = tol, axtol = tol, itmax = 10000, history = true)
    res_lsmr = stats_lsmr.residuals
    println("#### Stats-lsrn-cs ####")
    println("Résidus pour lsrn cs: ", norm(matrix * x - b))
    println("Norme de solution ", norm(x))
    println("Iterations: ", length(res))
    println("#### Stats-lsrn-lsqr ####")
    println("Résidus pour lsrn lsqr: ", norm(matrix * x_lsqr - b))
    println("Norme de solution ", norm(x_lsqr))
    println("Iterations: ", length(res_lsqr))
    println("#### Stats-lsqr ####")
    println("Résidus pour lsqr1: ", norm(matrix * x_lsqr1 - b))
    println("Norme de solution ", norm(x_lsqr1))
    println("Iterations: ", length(res_lsqr1))
    println("#### Stats-lsmr ####")
    println("Résidus pour lsmr: ", norm(matrix * x_lsmr - b))
    println("Norme de solution ", norm(x_lsmr))
    println("Iterations: ", length(res_lsmr))
    p = Plots.plot(1:length(res), log.(res), xlabel = "iter", ylabel = "residus", label = "LSRN-CS", legend = true)
    plot!(p, 1:length(res_lsqr), log.(res_lsqr), label = "LSRN-LSQR", legend = true)
    plot!(p, 1:length(res_lsqr1), log.(res_lsqr1), label = "LSQR", legend = true)
    plot!(p, 1:length(res_lsmr), log.(res_lsmr), label = "LSMR", legend = true)
    savefig(p, "res-"*name*".pdf")
end

solve_lsrn_l(landmark, "landmark")
solve_lsrn_r(rail4284, "rail4284")
solve_lsrn_r(rail582, "rail582")
solve_lsrn_l(specular, "specular")
solve_lsrn_r(stat96v1, "stat96v1")