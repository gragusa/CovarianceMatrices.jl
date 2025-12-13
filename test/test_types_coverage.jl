"""
Tests for type system coverage in CovarianceMatrices.jl.

This file tests constructors, accessors, and type behavior
that may not be covered by other test files.
"""

using Test
using CovarianceMatrices
using LinearAlgebra
using StatsAPI

@testset "Type System Coverage" begin

    @testset "HAC kernel constructors" begin
        # Type parameter syntax - Andrews
        @test Bartlett{Andrews}() isa HAC
        @test Parzen{Andrews}() isa HAC
        @test QuadraticSpectral{Andrews}() isa HAC
        @test TukeyHanning{Andrews}() isa HAC
        @test CovarianceMatrices.Truncated{Andrews}() isa HAC

        # Type parameter syntax - NeweyWest (only for some kernels)
        @test Bartlett{NeweyWest}() isa HAC
        @test Parzen{NeweyWest}() isa HAC
        @test QuadraticSpectral{NeweyWest}() isa HAC

        # NeweyWest should throw for TukeyHanning and Truncated
        @test_throws ArgumentError TukeyHanning{NeweyWest}()
        @test_throws ArgumentError CovarianceMatrices.Truncated{NeweyWest}()
        @test_throws ArgumentError TukeyHanning(NeweyWest)
        @test_throws ArgumentError CovarianceMatrices.Truncated(NeweyWest)

        # Function call syntax
        @test Bartlett(Andrews) isa HAC
        @test Parzen(NeweyWest) isa HAC
        @test QuadraticSpectral(Andrews) isa HAC

        # Fixed bandwidth
        @test Bartlett(5) isa HAC
        @test Parzen(3) isa HAC
        @test QuadraticSpectral(7) isa HAC
        @test TukeyHanning(4) isa HAC
        @test CovarianceMatrices.Truncated(6) isa HAC

        # Verify bandwidth is set
        k = Bartlett(10)
        @test k.bw[1] == 10.0

        k2 = Parzen(5)  # Fixed bandwidth via value
        @test k2.bw[1] == 5.0
    end

    @testset "VARHAC constructors and accessors" begin
        # Default constructor
        v = VARHAC()
        @test v.selector isa AICSelector
        @test v.strategy isa SameLags
        @test CovarianceMatrices.maxlags(v) == 8  # default

        # Symbol constructors
        @test VARHAC(:aic).selector isa AICSelector
        @test VARHAC(:bic).selector isa BICSelector
        @test_throws ArgumentError VARHAC(:invalid)

        # Integer constructor (max lags)
        @test VARHAC(12).strategy.maxlag == 12

        # FixedLags constructor
        vf = VARHAC(FixedLags(5))
        @test vf.selector isa FixedSelector
        @test vf.strategy.maxlag == 5

        # Full constructor
        v2 = VARHAC(BICSelector(), SameLags(10))
        @test v2.selector isa BICSelector
        @test v2.strategy isa SameLags
        @test CovarianceMatrices.maxlags(v2) == 10

        # Accessors before fitting
        @test CovarianceMatrices.AICs(v) === nothing
        @test CovarianceMatrices.BICs(v) === nothing
        @test CovarianceMatrices.order_aic(v) === nothing
        @test CovarianceMatrices.order_bic(v) === nothing

        # AutoLags
        v_auto = VARHAC(AICSelector(), AutoLags())
        @test_throws ErrorException CovarianceMatrices.maxlags(v_auto)  # Needs data dimensions
        @test CovarianceMatrices.maxlags(v_auto, 100, 2) > 0
        @test CovarianceMatrices.maxlags(v_auto, 1000, 5) > 0
    end

    @testset "Lag strategy constructors" begin
        # FixedLags
        @test FixedLags(5).maxlag == 5
        @test FixedLags(5.0).maxlag == 5
        @test FixedLags().maxlag == 5  # default

        # SameLags
        @test SameLags(10).maxlag == 10
        @test SameLags(10.0).maxlag == 10
        @test SameLags().maxlag == 8  # default

        # DifferentOwnLags
        @test DifferentOwnLags([3, 5]).maxlags == [3, 5]
        @test DifferentOwnLags((3, 5)).maxlags == [3, 5]
        @test DifferentOwnLags((3.0, 5.0)).maxlags == [3, 5]
        @test DifferentOwnLags().maxlags == [5, 5]  # default
    end

    @testset "EWC constructor" begin
        @test_throws ArgumentError EWC(0)
        @test_throws ArgumentError EWC(-1)
        @test_throws ArgumentError EWC(3.5)

        # Real that parses to Int should work
        @test EWC(3.0).B == 3
        @test EWC(5).B == 5
    end

    @testset "CR constructors" begin
        # Single vector
        g = [1, 1, 2, 2, 3, 3]
        @test CR0(g).g isa Tuple
        @test CR1(g).g isa Tuple
        @test CR2(g).g isa Tuple
        @test CR3(g).g isa Tuple

        # Symbol constructor
        @test CR0(:cluster).g isa Tuple{Vararg{Symbol}}
        @test CR1(:cluster).g isa Tuple{Vararg{Symbol}}

        # Multi-way clustering with varargs
        g1 = [1, 1, 2, 2]
        g2 = [1, 2, 1, 2]
        @test CR0(g1, g2).g isa Tuple
        @test CR1(g1, g2).g isa Tuple
        @test length(CR0(g1, g2).g) == 2

        # Multi-way with tuple
        @test CR0((g1, g2)).g isa Tuple
    end

    @testset "HC/HR type aliases" begin
        @test HC0 === HR0
        @test HC1 === HR1
        @test HC2 === HR2
        @test HC3 === HR3
        @test HC4 === HR4
        @test HC4m === HR4m
        @test HC5 === HR5
    end

    @testset "bandwidth accessor" begin
        k = Bartlett(5)
        @test CovarianceMatrices.bandwidth(k) == [5.0]

        k2 = Parzen{Andrews}()
        @test CovarianceMatrices.bandwidth(k2) == [0.0]  # Not yet computed
    end

    @testset "VarianceForm and Model types" begin
        @test Information() isa CovarianceMatrices.VarianceForm
        @test Misspecified() isa CovarianceMatrices.VarianceForm
        @test MLikeModel <: StatsAPI.StatisticalModel
        @test GMMLikeModel <: StatsAPI.StatisticalModel
    end

    @testset "Uncorrelated estimator" begin
        @test Uncorrelated() isa AbstractAsymptoticVarianceEstimator
    end

    @testset "LagSelector types" begin
        @test AICSelector() isa CovarianceMatrices.LagSelector
        @test BICSelector() isa CovarianceMatrices.LagSelector
        @test FixedSelector() isa CovarianceMatrices.LagSelector
    end

end
