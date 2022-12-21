## Test for CovarianceMatrices.jl
using CovarianceMatrices, DataFrames, CSV, Test, StableRNGs, CategoricalArrays

datadir = dirname(@__FILE__)
X = rand(StableRNG(123), 100, 3)

@testset "clustersum" begin        
    f = repeat(1:20, inner=5);
    𝐉 = CovarianceMatrices.clusterintervals(categorical(f))
    𝐉₀ = map(x->x:x+4, 1:5:100)
    @test collect(𝐉) == 𝐉₀
    M = CovarianceMatrices.clustersum(X, categorical(f))
    M₀= [134.8844  120.9909  123.9828
         120.9909  124.3984  120.7009
         123.9828  120.7009  127.6566]
    M .- M₀
    @test M ≈ M₀ atol=1e-4
end

@testset "Optimal Bandwidth (NeweyWest)" begin
    ## -- 
    𝒦 = Bartlett{NeweyWest}()
    Σ = a𝕍ar(𝒦, X)
    @test 𝒦.bw[1] ≈ 5.326955 atol=1e-6
    @test optimalbw(𝒦, X; prewhiten=false, demean=true) ≈ 𝒦.bw[1] rtol=1e-9

    𝒦 = Parzen{NeweyWest}()
    Σ = a𝕍ar(𝒦, X)
    @test 𝒦.bw[1] ≈ 9.72992 atol=1e-6
    @test optimalbw(𝒦, X; prewhiten=false, demean=true) ≈ 𝒦.bw[1] rtol=1e-9

    𝒦 = QuadraticSpectral{NeweyWest}()
    Σ = a𝕍ar(𝒦, X)
    @test 𝒦.bw[1] ≈ 4.833519 atol=1e-6
    @test optimalbw(𝒦, X; prewhiten=false, demean=true) ≈ 𝒦.bw[1] rtol=1e-9
    ## ---
    𝒦 = Bartlett{NeweyWest}()
    Σ = a𝕍ar(𝒦, X; prewhiten=true)
    @test 𝒦.bw[1] ≈ 1.946219 rtol=1e-7
    @test optimalbw(𝒦, X; prewhiten=true) == 𝒦.bw[1] 

    𝒦 = Parzen{NeweyWest}()
    Σ = a𝕍ar(𝒦, X; prewhiten=true)
    @test 𝒦.bw[1] ≈ 6.409343 rtol=1e-7
    @test optimalbw(𝒦, X; prewhiten=true) == 𝒦.bw[1]

    𝒦 = QuadraticSpectral{NeweyWest}()
    Σ = a𝕍ar(𝒦, X; prewhiten=true)
    @test 𝒦.bw[1] ≈ 3.183961 atol=1e-6
    @test optimalbw(𝒦, X; prewhiten=true) == 𝒦.bw[1]
end

@testset "Optimal Bandwidth (Andrews)" begin
    𝒦 = Bartlett{Andrews}()
    Σ = a𝕍ar(𝒦, X; prewhiten=true);
    @test 𝒦.bw[1] ≈ 0.3826726 atol=1e-5
    @test optimalbw(𝒦, X; prewhiten=true) ≈ 𝒦.bw[1] atol=1e-6

    𝒦 = Parzen{Andrews}()
    Σ = a𝕍ar(𝒦, X; prewhiten=true);
    @test 𝒦.bw[1] ≈ 1.377023 atol=1e-3
    𝒦 = Parzen{Andrews}()
    @test CovarianceMatrices.optimalbandwidth(𝒦, X; prewhiten=true) == 𝒦.bw[1] atol=1e-6

    𝒦 = QuadraticSpectral{Andrews}()
    Σ = a𝕍ar(𝒦, X; prewhiten=true)
    @test 𝒦.bw[1] ≈ 0.6840621 atol=1e-6
    @test optimalbandwidth(𝒦, X) == 𝒦.bw[1] atol=1e-6
    
    𝒦 = TukeyHanning{Andrews}()
    Σ = a𝕍ar(𝒦, X; prewhiten=true)
    𝒦.bw[1] ≈ 0.3420569 atol=1e-6
    @test optimalbandwidth(𝒦, X) == 𝒦.bw[1] atol=1e-6

    𝒦 = Truncated{Andrews}()
    Σ = a𝕍ar(𝒦, X; prewhiten=true)
    𝒦.bw[1] ≈ 0.3420569 atol=1e-6
    @test optimalbandwidth(𝒦, X) == 𝒦.bw[1] atol=1e-6

    𝒦 = Bartlett{Andrews}()
    Σ = a𝕍ar(𝒦, X; prewhiten=true)
    @test 𝒦.bw[1] ≈ 1.96508 atol=1e-6
    @test optimalbandwidth(𝒦, X) == 𝒦.bw[1] atol=1e-6

    𝒦 = Parzen{Andrews}()
    Σ = a𝕍ar(𝒦, X; prewhiten=true)
    @test 𝒦.bw[1] ≈ 6.433401 atol=1e-6
    @test optimalbandwidth(𝒦, X) == 𝒦.bw[1] atol=1e-6

    𝒦 = QuadraticSpectral{Andrews}()
    Σ = a𝕍ar(𝒦, X; prewhiten=true)
    @test 𝒦.bw[1] ≈ 3.195912 atol=1e-6
    @test optimalbandwidth(𝒦, X) == 𝒦.bw[1] atol=1e-6
    
    #𝒦 = TukeyHanning{Andrews}()
    #Σ = a𝕍ar(𝒦, X)
    #𝒦.bw[1] ≈ 4.833519

    #𝒦 = Truncated{Andrews}()
    #Σ = a𝕍ar(𝒦, X)
    #𝒦.bw[1] ≈ 4   

end


@testset "aVar" begin
    𝒦 = Bartlett{NeweyWest}()
    Σ = a𝕍ar(𝒦, X)
    Σ₀ = [0.1348844  0.1209909  0.1239828
          0.1209909  0.1243984  0.1207009
          0.1239828  0.1207009  0.1276566]
    @test Σ ≈ Σ₀ atol=1e-6

    𝒦 = Parzen{NeweyWest}()
    Σ = a𝕍ar(𝒦, X)
    Σ₀ = [0.1348844  0.1209909  0.1239828
          0.1209909  0.1243984  0.1207009
          0.1239828  0.1207009  0.1276566]
    @test Σ ≈ Σ₀ atol=1e-6

    𝒦 = QuadraticSpectral{NeweyWest}()
    Σ = a𝕍ar(𝒦, X)
    Σ₀ = [0.1348844  0.1209909  0.1239828
          0.1209909  0.1243984  0.1207009
          0.1239828  0.1207009  0.1276566]
    @test Σ ≈ Σ₀ atol=1e-6

    𝒦 = Bartlett{NeweyWest}()
    Σ = a𝕍ar(𝒦, X; prewhiten=true)
    Σ₀ = [0.1348844  0.1209909  0.1239828
          0.1209909  0.1243984  0.1207009
          0.1239828  0.1207009  0.1276566]
    @test Σ ≈ Σ₀ atol=1e-6

    𝒦 = Parzen{NeweyWest}()
    Σ = a𝕍ar(

@testset "clusterintervals" begin
    f = repeat(1:20, inner=5);
    
end


@testset "a𝕍ar CR()" begin
    X = CSV.File(joinpath(datadir, "testdata/X100x2.csv")) |> DataFrame |> Matrix;
    f = repeat(1:20, inner=5);
    Σ = a𝕍ar(CR0(f), X);
    Σ₀= [0.010003084285822686 0.002579249460680671
         0.002579249460680671 0.014440606823274103]
    @test Σ ≈ Σ₀


end