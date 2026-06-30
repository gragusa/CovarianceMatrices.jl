"""
Value-equality and hashing tests for estimator types.

Estimators are value objects: estimators with the same specification compare
equal and hash equally, and equality reflects the specification rather than
transient fit-state populated by `aVar`/`optimalbw`.
"""

using Test
using CovarianceMatrices
using Random

@testset "Estimator equality and hashing" begin
    @testset "Stateless markers" begin
        @test HC0() == HC0()
        @test HR0() == HC0()           # alias of the same type
        @test HC0() != HC1()
        @test Uncorrelated() == Uncorrelated()
        @test Andrews() == Andrews()
        @test Andrews() != NeweyWest()
        @test AICSelector() == AICSelector()
        @test AICSelector() != BICSelector()
        @test AutoLags() == AutoLags()
    end

    @testset "Field-bearing specifications" begin
        @test EWC(10) == EWC(10)
        @test EWC(10) != EWC(11)

        @test CR0([1, 1, 2, 2]) == CR0([1, 1, 2, 2])
        @test CR0([1, 1, 2, 2]) != CR0([1, 2, 2, 2])
        @test CR0([1, 1, 2, 2]) != CR1([1, 1, 2, 2])
        @test CR0((:firm, :year)) == CR0((:firm, :year))
        @test CR0((:firm, :year)) != CR0((:firm,))

        @test SameLags(8) == SameLags(8)
        @test SameLags(8) != SameLags(9)
        @test DifferentOwnLags([3, 5]) == DifferentOwnLags([3, 5])
        @test DifferentOwnLags([3, 5]) != DifferentOwnLags([3, 6])

        @test UniformSmoother(5) == UniformSmoother(5)
        @test UniformSmoother(5) != UniformSmoother(6)
        @test UniformSmoother(5) != TriangularSmoother(5)
    end

    @testset "HAC equality reflects the specification, not fit-state" begin
        @test Bartlett(2) == Bartlett(2)
        @test Bartlett(2) != Bartlett(3)
        @test Bartlett{Andrews}() == Bartlett{Andrews}()
        @test Bartlett{Andrews}() != Parzen{Andrews}()
        @test Bartlett{Andrews}() != Bartlett{NeweyWest}()
        @test Bartlett(2) != Bartlett{Andrews}()

        # A fitted estimator stays equal to its unfitted self: fitting populates
        # bw/kw/wlock, which equality ignores.
        Random.seed!(1)
        X = randn(200, 3)
        fitted = Bartlett{Andrews}()
        unfitted = Bartlett{Andrews}()
        aVar(fitted, X)
        # fit-state did change (bandwidth is an unexported accessor)
        @test CovarianceMatrices.bandwidth(fitted) != CovarianceMatrices.bandwidth(unfitted)
        @test fitted == unfitted
        @test hash(fitted) == hash(unfitted)
    end

    @testset "VARHAC equality reflects selector and strategy only" begin
        @test VARHAC() == VARHAC()
        @test VARHAC(:aic) != VARHAC(:bic)
        @test VARHAC(AICSelector(), SameLags(8)) != VARHAC(AICSelector(), SameLags(12))

        Random.seed!(2)
        X = randn(200, 3)
        fitted = VARHAC(AICSelector(), SameLags(4))
        unfitted = VARHAC(AICSelector(), SameLags(4))
        aVar(fitted, X)
        # fit-state did change (order_aic is an unexported accessor)
        @test CovarianceMatrices.order_aic(fitted) !== nothing
        @test CovarianceMatrices.order_aic(unfitted) === nothing
        @test fitted == unfitted
        @test hash(fitted) == hash(unfitted)
    end

    @testset "hash is consistent with ==" begin
        pairs = Any[
            (HC0(), HC0()),
            (EWC(10), EWC(10)),
            (CR0([1, 1, 2, 2]), CR0([1, 1, 2, 2])),
            (CR0((:firm, :year)), CR0((:firm, :year))),
            (DifferentOwnLags([3, 5]), DifferentOwnLags([3, 5])),
            (UniformSmoother(5), UniformSmoother(5)),
            (Bartlett(2), Bartlett(2)),
            (Bartlett{Andrews}(), Bartlett{Andrews}()),
            (VARHAC(), VARHAC()),
        ]
        for (a, b) in pairs
            @test a == b
            @test hash(a) == hash(b)
        end
    end

    @testset "Usable as Dict keys and unique elements" begin
        d = Dict(HC0() => 1, CR0([1, 1, 2, 2]) => 2, Bartlett(2) => 3)
        @test d[HC0()] == 1
        @test d[CR0([1, 1, 2, 2])] == 2
        @test d[Bartlett(2)] == 3

        @test length(unique([Bartlett(2), Bartlett(2), Bartlett(3), Bartlett{Andrews}()])) == 3
        @test length(unique([VARHAC(), VARHAC(), VARHAC(:bic)])) == 2
        @test length(unique([CR0([1, 1, 2, 2]), CR0([1, 1, 2, 2]), CR1([1, 1, 2, 2])])) == 2
    end
end
