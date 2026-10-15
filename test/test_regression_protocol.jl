## A `RegressionModel` that implements only the documented protocol gets the same
## variances as a GLM fit of the same regression.

using Test
using CovarianceMatrices
using LinearAlgebra
using DataFrames
using GLM
using StableRNGs
using StatsAPI
using StatsBase: StatsBase

const CMRP = CovarianceMatrices

struct ProtocolOLS <: StatsBase.RegressionModel
    X::Matrix{Float64}
    y::Vector{Float64}
    β::Vector{Float64}
end
ProtocolOLS(X, y) = ProtocolOLS(X, y, X \ y)
StatsAPI.coef(m::ProtocolOLS) = m.β
StatsAPI.modelmatrix(m::ProtocolOLS) = m.X
StatsAPI.residuals(m::ProtocolOLS) = m.y - m.X * m.β
StatsAPI.leverage(m::ProtocolOLS) = diag(m.X * inv(m.X' * m.X) * m.X')
StatsAPI.weights(m::ProtocolOLS) = Float64[]
CMRP.bread(m::ProtocolOLS) = inv(m.X' * m.X)

# Implements the required methods and nothing optional.
struct MinimalOLS <: StatsBase.RegressionModel
    X::Matrix{Float64}
    y::Vector{Float64}
    β::Vector{Float64}
end
MinimalOLS(X, y) = MinimalOLS(X, y, X \ y)
StatsAPI.coef(m::MinimalOLS) = m.β
StatsAPI.modelmatrix(m::MinimalOLS) = m.X
CMRP.momentresiduals(m::MinimalOLS) = m.y - m.X * m.β
CMRP.bread(m::MinimalOLS) = inv(m.X' * m.X)

struct ProtocolMLE <: MLikeModel
    Z::Matrix{Float64}
    H::Matrix{Float64}
end
StatsAPI.coef(m::ProtocolMLE) = zeros(size(m.Z, 2))
StatsAPI.nobs(m::ProtocolMLE) = size(m.Z, 1)
CMRP.momentmatrix(m::ProtocolMLE) = m.Z
CMRP.hessian_objective(m::ProtocolMLE) = m.H

@testset "Regression-model protocol" begin
    rng = StableRNG(20261015)
    n = 80
    df = DataFrame(x = randn(rng, n), g = repeat(1:8, inner = 10), h = repeat(1:5, 16))
    df.y = 1 .+ 0.5 .* df.x .+ randn(rng, n)
    glmfit = lm(@formula(y ~ x), df)
    X = modelmatrix(glmfit)
    ols = ProtocolOLS(X, df.y)

    @testset "protocol names are public" begin
        for s in (:bread, :momentresiduals, :leverage, :mask)
            @test Base.ispublic(CovarianceMatrices, s)
            @test !Base.isexported(CovarianceMatrices, s)
        end
    end

    @testset "fallbacks" begin
        @test CMRP.momentresiduals(ols) ≈ residuals(glmfit)
        @test CMRP.leverage(ols) ≈ CMRP.leverage(glmfit)
        @test CMRP.mask(ols) == [true, true]
    end

    @testset "matches GLM: $(nameof(typeof(k)))" for k in (
        HC0(), HC1(), HC2(), HC3(), HC4(), HC4m(), HC5(), Bartlett(3),
        Bartlett{Andrews}(), Parzen{NeweyWest}(), EWC(4), VARHAC())
        V = vcov(k, ols)
        @test V isa CovarianceMatrix
        @test V ≈ vcov(k, glmfit)
        @test stderror(k, ols) ≈ stderror(k, glmfit)
    end

    @testset "matches GLM: $(nameof(typeof(k)))" for k in (
        CR0(df.g), CR1(df.g), CR2(df.g), CR3(df.g), CR1((df.g, df.h)))
        @test vcov(k, ols) ≈ vcov(k, glmfit)
    end

    @testset "model with only the required methods" begin
        minimal = MinimalOLS(X, df.y)
        @test vcov(HC1(), minimal) ≈ vcov(HC1(), glmfit)
        @test vcov(CR1(df.g), minimal) ≈ vcov(CR1(df.g), glmfit)
        # The leverage-based estimators need `leverage`.
        @test_throws MethodError vcov(HC3(), minimal)
        @test !isdefined(CovarianceMatrices, :_residuals)
    end

    @testset "GLM fits use the weighted working residuals" begin
        w = 0.5 .+ rand(rng, n)
        wfit = lm(@formula(y ~ x), df; wts = w)
        @test CMRP.momentresiduals(wfit) ≈ w .* residuals(wfit)
        @test CMRP.numobs(wfit) == n
    end

    @testset "vcov(Information(), model) for an MLikeModel" begin
        Z = randn(rng, n, 2)
        mle = ProtocolMLE(Z, Z' * Z)
        V = vcov(Information(), mle)
        @test V isa CovarianceMatrix
        @test estimator(V) == Information()
        @test V ≈ inv(Z' * Z)
        @test stderror(Information(), mle) ≈ sqrt.(diag(V))
        @test_throws MethodError vcov(HC0(), Information(), mle)
    end
end
