using Plots
using SparseArrays

include("../src/LSRN.jl")

Random.seed!(1234)

# Test LSRN_l with big and sparse matrix


function solve_lsrn_l(matrix, name)
    tol = 1e-10
    x_true = rand(size(matrix, 2))
    b = matrix * x_true
    x, res, _ = LSRN_l(matrix, b, γ = 1.5)
    x_lsqr, res_lsqr, _ = LSRN_l(matrix, b, γ = 1.5, subsolver = :LSQR)
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
    x, res, _ = LSRN_r(matrix, b, γ = 1.5)
    x_lsqr, res_lsqr, _ = LSRN_r(matrix, b, γ = 1.5, subsolver = :LSQR)
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
m = 1000000
n = 1000
x = rand(n)
A = sprand(m, n, 0.001)
b = A * x
solve_lsrn_l(A, "sparse-left")
x = rand(m)
Ar = sprand(n, m, 0.001)
b = Ar * x
solve_lsrn_r(Ar, "sparse-right")
