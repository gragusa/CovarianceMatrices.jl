## Test for CovarianceMatrices.jl
using CovarianceMatrices, DataFrames, CSV, Test, StableRNGs, CategoricalArrays, Statistics, LinearAlgebra
using JSON

const CM = CovarianceMatrices

datadir = dirname(@__FILE__)
X = rand(StableRNG(123), 100, 3)
Y = rand(StableRNG(123), 100)
df = DataFrame(X, :auto)
df.y = Y

@testset "demean" begin
    @test demean(X; dims = 1) == X .- mean(X; dims=1)
    @test mean(demean(X; dims = 1); dims=1) == mean(X .- mean(X; dims=1); dims = 1)

    @test demean(X'; dims = 2) == X' .- mean(X'; dims=2)
    @test mean(demean(X'; dims = 2); dims=2) == mean(X' .- mean(X'; dims=2); dims = 2)

    m = mean(X; dims = 1)
    @test demean(X; dims=1, means=m) == demean(X; dims=1)

    m = mean(X; dims = 2)
    @test demean(X; dims=2, means=m) == demean(X; dims=2)
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
    Σ = a𝕍ar(𝒦, X; prewhiten=false);
    @test 𝒦.bw[1] ≈ 2.329739 rtol=1e-6
    @test optimalbw(𝒦, X; prewhiten=false) == 𝒦.bw[1] 

    𝒦 = Parzen{Andrews}()
    Σ = a𝕍ar(𝒦, X; prewhiten=false);
    @test 𝒦.bw[1] ≈ 4.81931 rtol=1e-6
    @test CovarianceMatrices.optimalbw(𝒦, X; prewhiten=false) == 𝒦.bw[1]

    𝒦 = QuadraticSpectral{Andrews}()
    Σ = a𝕍ar(𝒦, X; prewhiten=false)
    @test 𝒦.bw[1] ≈ 2.394082 atol=1e-6
    @test optimalbw(𝒦, X) == 𝒦.bw[1]
    
    𝒦 = TukeyHanning{Andrews}()
    Σ = a𝕍ar(𝒦, X; prewhiten=false)
    @test 𝒦.bw[1] ≈ 3.162049 rtol=1e-6
    @test optimalbw(𝒦, X) == 𝒦.bw[1]

    𝒦 = Truncated{Andrews}()
    Σ = a𝕍ar(𝒦, X; prewhiten=false)
    @test 𝒦.bw[1] ≈ 1.197131 rtol=1e-6
    @test optimalbw(𝒦, X) == 𝒦.bw[1]

    ## --
    𝒦 = Bartlett{Andrews}()
    Σ = a𝕍ar(𝒦, X; prewhiten=true);
    @test 𝒦.bw[1] ≈ 0.3836096 rtol=1e-6
    @test optimalbw(𝒦, X; prewhiten=true) == 𝒦.bw[1] 

    𝒦 = Parzen{Andrews}()
    Σ = a𝕍ar(𝒦, X; prewhiten=true);
    @test 𝒦.bw[1] ≈ 1.380593 rtol=1e-6
    @test CovarianceMatrices.optimalbw(𝒦, X; prewhiten=true) == 𝒦.bw[1]

    𝒦 = QuadraticSpectral{Andrews}()
    Σ = a𝕍ar(𝒦, X; prewhiten=true)
    @test 𝒦.bw[1] ≈ 0.6858351 atol=1e-6
    @test optimalbw(𝒦, X) == 𝒦.bw[1]
    
    𝒦 = TukeyHanning{Andrews}()
    Σ = a𝕍ar(𝒦, X; prewhiten=true)
    @test 𝒦.bw[1] ≈ 0.9058356 rtol=1e-6
    @test optimalbw(𝒦, X) == 𝒦.bw[1]

    𝒦 = Truncated{Andrews}()
    Σ = a𝕍ar(𝒦, X; prewhiten=true)
    @test 𝒦.bw[1] ≈ 0.3429435 rtol=1e-6
    @test optimalbw(𝒦, X) == 𝒦.bw[1]

end

@testset "clustersum" begin        
  f = repeat(1:20, inner=5);
  𝐉 = CovarianceMatrices.clusterintervals(categorical(f))
  𝐉₀ = map(x->x:x+4, 1:5:100)
  @test collect(𝐉) == 𝐉₀
  M = CovarianceMatrices.clustersum(X, categorical(f))
  M₀= [134.8844  120.9909  123.9828
       120.9909  124.3984  120.7009
       123.9828  120.7009  127.6566]    
  @test M ≈ M₀ atol=1e-4
end

Σ₀₀ = [
       [0.07978827089743976 0.005560324649425609 0.009703799309186547
        0.005560324649425609 0.08276474874770062 0.0010530436728554352
        0.009703799309186547 0.0010530436728554352 0.07431486592263496],
    
        [0.08824047441166522 0.01185475583687493 0.014226486564055717
        0.01185475583687493 0.08753222220168377 0.005137675013885958
        0.014226486564055717 0.005137675013885958 0.07382524667188076],
    
        [0.08304722971511129 0.009566557076336682 0.015084732026814018
        0.009566557076336682 0.08487958325675327 0.0033678564195238933
        0.015084732026814018 0.0033678564195238933 0.07782361388868414],
    
        [0.08559572719755348 0.010100014851936187 0.014007312794571108
        0.010100014851936187 0.0854452011439353 0.003987375876407169
        0.014007312794571108 0.003987375876407169 0.07530458218249476],

        [0.09113048508616475 0.01012761283802301 0.014949485990870565
        0.01012761283802301 0.08417657485409089 0.005652466376470253
        0.014949485990870565 0.005652466376470253 0.07567045699291326],
    
        [0.10403285139514229 0.017161666552281015 0.012937499243192903
        0.017161666552281015 0.09461953256288745 0.009862502341427618
        0.012937499243192903 0.009862502341427618 0.06415361805073096],
    
        [0.11919732774525865 0.022941382516275952 0.015095058906094373
        0.022941382516275952 0.10152132051735772 0.012217515924874448
        0.015095058906094373 0.012217515924874448 0.060401394896807015],
    
        [0.11286141828969598 0.020980921344488233 0.01823291328076095
        0.020980921344488233 0.09941184965586379 0.015319761791783473
        0.01823291328076095 0.015319761791783473 0.06292735773558823],
    
        [0.10214970819829594 0.01795183790594208 0.023765383078499024
        0.017951837905942077 0.08626897790931316 0.009428159436790295
        0.023765383078499024 0.009428159436790299 0.07941946726668297],
    
        [0.10203584285781796 0.01799717571386372 0.02359634430350738
        0.017997175713863715 0.08633822583613252 0.00940840735139296
        0.02359634430350738 0.009408407351392964 0.07932583171604031],
    
        [0.1027070205998854 0.01785906774640339 0.024081614850023098
        0.017859067746403393 0.08624976234548798 0.0095727497846355
        0.024081614850023098 0.009572749784635497 0.07940007875065369],
    
        [0.10214970819829594 0.01795183790594208 0.023765383078499024
        0.017951837905942077 0.08626897790931316 0.009428159436790295
        0.023765383078499024 0.009428159436790299 0.07941946726668297],
    
        [0.10214970819829594 0.01795183790594208 0.023765383078499024
        0.017951837905942077 0.08626897790931316 0.009428159436790295
        0.023765383078499024 0.009428159436790299 0.07941946726668297],
    
        [0.10082848215180824 0.018477911006547318 0.021803957139334383
        0.01847791100654732 0.0870724898840273 0.009198967923465359
        0.021803957139334383 0.009198967923465357 0.0783329757332015],
    
        [0.10319135331171438 0.02444221028323441 0.01847305161846699
        0.02444221028323441 0.09549523073371463 0.012448284901890937
        0.01847305161846699 0.012448284901890937 0.07093488602565269],

        [0.09310133199441366 0.02187370477388356 0.01597340791678698
        0.021873704773883562 0.09162619720253418 0.009583822285400886
        0.015973407916786978 0.009583822285400884 0.07403525954398654]
        
        ]

kernels = (Bartlett{Andrews}(), 
            Parzen{Andrews}(), 
            QuadraticSpectral{Andrews}(), 
            TukeyHanning{Andrews}(), 
            Truncated{Andrews}(), 
            Bartlett{NeweyWest}(), 
            Parzen{NeweyWest}(), 
            QuadraticSpectral{NeweyWest}())

pre = (false, true)            

@testset "aVar HAC" begin
    for ((𝒦, prewhiten), Σ₀) in zip(Iterators.product(kernels, pre), Σ₀₀)
        Σ = a𝕍ar(𝒦, X; prewhiten=prewhiten)
        @test Σ ≈ Σ₀ rtol=1e-6
    end
end

kernels = (HR0(), HR1(), HR2(), HR3(), HR4(), HR4m(), HR5())

Σ₀ = [ 0.06415873470586395 -0.004015858202035743 -6.709834054283887e-5; 
      -0.004015858202035743 0.07800552644879759 -0.00615707811722861; 
      -6.709834054283887e-5 -0.00615707811722861 0.07184846516936118]


@testset "aVar HRx" begin
        for 𝒦 in kernels
            Σ = a𝕍ar(𝒦, X)
            @test Σ ≈ Σ₀ rtol=1e-6
        end
    end
    




@testset "CRHC............................................." begin        
    cl = repeat(1:5, inner=20)
    𝒦 = CR0(cl)
    Σ = a𝕍ar(𝒦, X)
    ## sandwich package uses HC0 but always scales by G/(G-1)
    ## below is the result of 
    ## v = vcovCL(lm(X~1), cl, type="HC0")
    Σ₀ =  [0.0013477493837805246 0.0001411950613289987 4.6345925014758175e-5; 
           0.0001411950613289987 0.0004985361461159058 -0.00039126414097571385; 
           4.6345925014758175e-5 -0.00039126414097571385 0.00033308110226548546]
    
    @test Σ./(5-1) ≈ Σ₀ rtol = 1e-8
    ## Since a𝕍ar is scaled by (G/n^2), this is equivalent to  dividing by (1/G) to get the
    ## standard error and then multiply by G/(G-1) to apply the correction.

    ## Out of order clusters
    cl = repeat(1:5, outer=20)
    𝒦 = CR0(cl)
    Σ = a𝕍ar(𝒦, X)
end

@testset "Some king...................................." begin
    Z = round.(Int, X*100)
    Σ = a𝕍ar(Bartlett{Andrews}(), Z; prewhiten=false)
end


@testset "Driscol and Kraay"
  df = CSV.read(joinpath(datadir,"testdata/grunfeld.csv"), DataFrame)
  #df = RDatasets.dataset("Ecdat", "Grunfeld")
  X = [ones(size(df,1)) df.Value df.Capital]
  y = df.Inv
  β = X\y
  ## Moment Matrix
  m = X.*(y .- X*β)
  ## Driscol Kraay Variance Covariance Matrix
  T = length(unique(df.Year))
  bw = 5
  𝒦 = CovarianceMatrices.DriscollKraay(df.Year, df.Firm, Bartlett(bw))
  Σ = a𝕍ar(𝒦, m; scale=false)
  F = inv(cholesky(X'X))
  Σ₀ = F*Σ*F.*T
  #library(Ecdat)
  #library(plm)
  #data("Grunfeld")
  #zz <- plm(inv~value+capital, data=Grunfeld, model = "pooling")
  #vcovSCC(zz, maxlag = 4, )
  # Note: maxlag = 4 is equivalent to bw = 5
  Σ_ssc = [148.60459392965311     -0.067282610486606179   -0.32394796987915847;
            -0.067282610486606151  0.00018052654961234828 -0.00035661048571690061; 
            -0.32394796987915825  -0.00035661048571690066  0.0024312798435615107]
  @test Σ₀ ≈ Σ_ssc rtol=1e-6
end


## Test GLM Interface
const andrews_kernels = [:Truncated, :Parzen, :TukeyHanning, :QuadraticSpectral, :Bartlett]
const neweywest_kernels = [:Parzen, :QuadraticSpectral, :Bartlett]

using GLM

function Base.String(k::Type{<:CovarianceMatrices.})
    return string(k)
end

reg = JSON.parse(read(joinpath(datadir, "testdata/regression.json"), String))
df = CSV.File(joinpath(datadir, "testdata/ols_df.csv")) |> DataFrame

df2 = CSV.File(joinpath("/home/gragusa", "df.csv")) |> DataFrame
function fopt!(u)
    global da = Dict{String, Any}()
    global dn = Dict{String, Any}()
    for pre in (:false, :true)
        da["bwtype"] = "auto"
        da["prewhite"] = pre == :true ? true : false
        dn["bwtype"] = "auto"
        dn["prewhite"] = pre == :true ? true : false
        
        for k in andrews_kernels
            eval(quote
                 ols = glm(@formula(y~x1+x2+x3), $df, Normal(), IdentityLink())
                 𝒦 = ($k){Andrews}()
                 𝒦.weights = [1,0.0,0.0]
                 tmp = vcov(𝒦, ols; prewhiten=$pre)
                 da[String($k)] = Dict{String, Any}("bw" => CM.bandwidth(𝒦), "V" => tmp)
          end)
        end
        
        for k in neweywest_kernels
          eval(quote
            𝒦 = ($k){NeweyWest}()
            ## To get the same results of R, the weights given to the intercept should be 0
            𝒦.weights = [0.0,1.0,1.0,1.0]
            tmp = vcov(𝒦, ols; prewhiten=$pre)
            dn[String($k)] = Dict{String, Any}("bw" => CM.bandwidth(𝒦), "V" => tmp)
          end)
        end
        push!(u, Dict("andrews" => da, "neweywest" => dn))
        da = Dict{String, Any}()
        dn = Dict{String, Any}()
    end
end

function ffix!(u)
  global da = Dict{String, Any}()
  global dn = Dict{String, Any}()
  for pre in (:false, :true)
      da["bwtype"] = "auto"
      da["prewhite"] = pre == :true ? true : false
      dn["bwtype"] = "auto"
      dn["prewhite"] = pre == :true ? true : false
      for k in andrews_kernels
          eval(quote
               ols = glm(@formula(y~x1+x2+x3), $df, Normal(), IdentityLink())
               𝒦 = ($k)(1.5)
               𝒦.weights = [1,0.0,0.0]
               tmp = vcov(𝒦, ols; prewhiten=$pre)
               da[String($k)] = Dict{String, Any}("bw" => CM.bandwidth(𝒦), "V" => tmp.V)
               if Symbol($k) in neweywest_kernels
                  𝒦 = ($k)(1.5)
                  tmp = vcov(𝒦, ols; prewhiten=$pre)
                  dn[String($k)] = Dict{String, Any}("bw" => CM.bandwidth(𝒦), "V" => tmp.V)
               end
               end)
      end
      push!(u, Dict("andrews" => da, "neweywest" => dn))
      da = Dict{String, Any}()
      dn = Dict{String, Any}()
  end
end

u = Any[]
fopt!(u)

for j in 1:4, h in ("andrews",), k in ("Truncated", "Bartlett", "Tukey-Hanning", "Quadratic Spectral")
  @test hcat(reg[j][h][k]["V"]...) ≈ u[j][h][k]["V"]
  @test reg[j][h][k]["bw"] ≈ u[j][h][k]["bw"]
end






