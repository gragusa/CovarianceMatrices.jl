## Estimators on element types other than Float64 and on sparse input.

using Test
using CovarianceMatrices
using ForwardDiff
using SparseArrays
using LinearAlgebra
using StableRNGs

@testset "Generic element types and sparse input" begin
    rng = StableRNG(2024)
    n = 120
    X = randn(rng, n, 3)
    y = X * [1.0, 2.0, 3.0] / 10 + randn(rng, n)
    tis = repeat(1:30, inner = 4)
    iis = repeat(1:4, outer = 30)
    grp = repeat(1:12, inner = 10)
    moments(θ) = X .* (y .- X * θ)
    θ0 = [0.1, 0.2, 0.3]

    families = [
        HC0(),
        CR0(grp),
        CovarianceMatrices.Cluster(grp),
        Bartlett(4),
        Parzen(3),
        QuadraticSpectral(2.5),
        Bartlett{Andrews}(),
        Parzen{NeweyWest}(),
        EWC(5),
        VARHAC(),
        VARHAC(FixedLags(2)),
        VARHAC(BICSelector(), DifferentOwnLags([2, 1])),
        UniformSmoother(4),
        TriangularSmoother(4),
        DriscollKraay(Bartlett(3), tis = tis, iis = iis)
    ]

    @testset "ForwardDiff through aVar: $(nameof(typeof(k))), prewhite = $pw" for k in families,
        pw in (false, true)

        CovarianceMatrices._prewhitens(k) || !pw || continue
        f(θ) = sum(aVar(k, moments(θ); prewhite = pw))

        # The value part of a Dual evaluation equals the Float64 result.
        Vd = aVar(k, ForwardDiff.Dual.(moments(θ0), 1.0); prewhite = pw)
        @test ForwardDiff.value.(parent(Vd)) ≈ aVar(k, moments(θ0); prewhite = pw)

        # Gradients agree with central finite differences.
        gad = ForwardDiff.gradient(f, θ0)
        h = 1e-6
        gfd = [(f(θ0 + h * e) - f(θ0 - h * e)) / (2h) for e in eachcol(Matrix(1.0I, 3, 3))]
        @test gad ≈ gfd rtol = 1e-5
    end

    @testset "sparse input agrees with dense" begin
        S = sprand(rng, 200, 4, 0.3)
        D = Matrix(S)
        g = rand(rng, 1:15, 200)
        g2 = rand(rng, 1:6, 200)

        cl = CovarianceMatrices.Clustering(g)
        @test CovarianceMatrices.clusterize(S, cl) ≈ CovarianceMatrices.clusterize(D, cl)
        E_s, B_s = CovarianceMatrices.fit_var(S)
        E_d, B_d = CovarianceMatrices.fit_var(D)
        @test E_s ≈ E_d
        @test B_s ≈ B_d

        for k in (CR0(g), CR0((g, g2)), CovarianceMatrices.Cluster(g), HC0(), Bartlett(3))
            @test aVar(k, S) ≈ aVar(k, D)
            @test aVar(k, S; demean = false) ≈ aVar(k, D; demean = false)
        end
        @test aVar(Bartlett(3), S; demean = false, prewhite = true) ≈
              aVar(Bartlett(3), D; demean = false, prewhite = true)
    end

    @testset "cluster estimators on views" begin
        D = randn(rng, 60, 3)
        g = repeat(1:6, inner = 10)
        W = randn(rng, 80, 3)
        W[1:60, :] .= D
        for k in (CR0(g), CR0((g, repeat(1:10, outer = 6))))
            @test aVar(k, view(W, 1:60, :)) ≈ aVar(k, D)
            @test aVar(k, view(W, 1:60, :); demean = false) ≈ aVar(k, D; demean = false)
        end
    end
end
