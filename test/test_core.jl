## Core functionality tests for CovarianceMatrices.jl
using CovarianceMatrices, DataFrames, CSV, Test, Random, StableRNGs, Statistics, LinearAlgebra, GroupedArrays
using JSON
using GLM

const CM = CovarianceMatrices

# Test data setup
datadir = dirname(@__FILE__)
X = rand(StableRNG(123), 100, 3)
Y = rand(StableRNG(123), 100)
df = DataFrame(X, :auto)
df.y = Y;

@testset "Core Functionality Tests" begin

@testset "Demeaning Operations ✅" begin
  @test CM.demeaner(X; dims = 1) == X .- mean(X; dims=1)
  @test mean(CM.demeaner(X; dims = 1); dims=1) == mean(X .- mean(X; dims=1); dims = 1)
  @test CM.demeaner(X'; dims = 2) == (X' .- mean(X'; dims=2))'
  m = mean(X; dims = 1)
  @test CM.demeaner(X; dims=1, means=m) == CM.demeaner(X; dims=1)
  m = mean(X; dims = 2)
  @test CM.demeaner(X; dims=2, means=m) == CM.demeaner(X; dims=2)
end

@testset "Optimal Bandwidth - NeweyWest ✅" begin
  ## --
  𝒦 = Bartlett{NeweyWest}()
  Σ = a𝕍ar(𝒦, X)
  @test 𝒦.bw[1] ≈ 5.326955 atol=1e-6
  @test optimalbw(𝒦, X; prewhite=false, demean=true) ≈ 𝒦.bw[1] rtol=1e-9

  𝒦 = Parzen{NeweyWest}()
  Σ = a𝕍ar(𝒦, X)
  @test 𝒦.bw[1] ≈ 9.72992 atol=1e-6
  @test optimalbw(𝒦, X; prewhite=false, demean=true) ≈ 𝒦.bw[1] rtol=1e-9

  𝒦 = QuadraticSpectral{NeweyWest}()
  Σ = a𝕍ar(𝒦, X)
  @test 𝒦.bw[1] ≈ 4.833519 atol=1e-6
  @test optimalbw(𝒦, X; prewhite=false, demean=true) ≈ 𝒦.bw[1] rtol=1e-9
  ## ---
  𝒦 = Bartlett{NeweyWest}()
  Σ = a𝕍ar(𝒦, X; prewhite=true)
  @test 𝒦.bw[1] ≈ 1.946219 rtol=1e-7
  @test optimalbw(𝒦, X; prewhite=true) == 𝒦.bw[1]

  𝒦 = Parzen{NeweyWest}()
  Σ = a𝕍ar(𝒦, X; prewhite=true)
  @test 𝒦.bw[1] ≈ 6.409343 rtol=1e-7
  @test optimalbw(𝒦, X; prewhite=true) == 𝒦.bw[1]

  𝒦 = QuadraticSpectral{NeweyWest}()
  Σ = a𝕍ar(𝒦, X; prewhite=true)
  @test 𝒦.bw[1] ≈ 3.183961 atol=1e-6
  @test optimalbw(𝒦, X; prewhite=true) == 𝒦.bw[1]
end

@testset "Optimal Bandwidth - Andrews ✅" begin
  𝒦 = Bartlett{Andrews}()
  Σ = a𝕍ar(𝒦, X; prewhite=false);
  @test 𝒦.bw[1] ≈ 2.329739 rtol=1e-6
  @test optimalbw(𝒦, X; prewhite=false) == 𝒦.bw[1]

  𝒦 = Parzen{Andrews}()
  Σ = a𝕍ar(𝒦, X; prewhite=false);
  @test 𝒦.bw[1] ≈ 4.81931 rtol=1e-6
  @test CovarianceMatrices.optimalbw(𝒦, X; prewhite=false) == 𝒦.bw[1]

  𝒦 = QuadraticSpectral{Andrews}()
  Σ = a𝕍ar(𝒦, X; prewhite=false)
  @test 𝒦.bw[1] ≈ 2.394082 atol=1e-6
  @test optimalbw(𝒦, X) == 𝒦.bw[1]

  𝒦 = TukeyHanning{Andrews}()
  Σ = a𝕍ar(𝒦, X; prewhite=false)
  @test 𝒦.bw[1] ≈ 3.162049 rtol=1e-6
  @test optimalbw(𝒦, X) == 𝒦.bw[1]

  𝒦 = Truncated{Andrews}()
  Σ = a𝕍ar(𝒦, X; prewhite=false)
  @test 𝒦.bw[1] ≈ 1.197131 rtol=1e-6
  @test optimalbw(𝒦, X) == 𝒦.bw[1]

  ## --
  𝒦 = Bartlett{Andrews}()
  Σ = a𝕍ar(𝒦, X; prewhite=true);
  @test 𝒦.bw[1] ≈ 0.3836096 rtol=1e-6
  @test optimalbw(𝒦, X; prewhite=true) == 𝒦.bw[1]

  𝒦 = Parzen{Andrews}()
  Σ = a𝕍ar(𝒦, X; prewhite=true);
  @test 𝒦.bw[1] ≈ 1.380593 rtol=1e-6
  @test CovarianceMatrices.optimalbw(𝒦, X; prewhite=true) == 𝒦.bw[1]

  𝒦 = QuadraticSpectral{Andrews}()
  Σ = a𝕍ar(𝒦, X; prewhite=true)
  @test 𝒦.bw[1] ≈ 0.6858351 atol=1e-6
  @test optimalbw(𝒦, X) == 𝒦.bw[1]

  𝒦 = TukeyHanning{Andrews}()
  Σ = a𝕍ar(𝒦, X; prewhite=true)
  @test 𝒦.bw[1] ≈ 0.9058356 rtol=1e-6
  @test optimalbw(𝒦, X) == 𝒦.bw[1]

  𝒦 = Truncated{Andrews}()
  Σ = a𝕍ar(𝒦, X; prewhite=true)
  @test 𝒦.bw[1] ≈ 0.3429435 rtol=1e-6
  @test optimalbw(𝒦, X) == 𝒦.bw[1]
end

@testset "Cold Start Bandwidth ✅" begin
    𝒦 = Bartlett{Andrews}()
    @test optimalbw(𝒦, X; prewhite=false) ≈ 2.329739 rtol=1e-6
    @test optimalbw(𝒦, X; prewhite=false) ≈ 2.329739 rtol=1e-6
end

@testset "Cluster Sum Operations ✅" begin
  f = repeat(1:20, inner=5);
  M = CovarianceMatrices.clusterize(X, GroupedArray(f))
  M₀= [134.8844  120.9909  123.9828
       120.9909  124.3984  120.7009
       123.9828  120.7009  127.6566]
  @test M ≈ M₀ atol=1e-4
  ## Out of order
  shuffler = shuffle(StableRNG(123), 1:size(X,1))
  Xo = X[shuffler, :]
  fo = f[shuffler]
  Mo = CovarianceMatrices.clusterize(Xo, GroupedArray(fo))
  @test Mo ≈ M
end

# Reference values for HAC kernels
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

@testset "HAC Variance Computations ✅" begin
  for ((𝒦, prewhite), Σ₀) in zip(Iterators.product(kernels, pre), Σ₀₀)
    Σ = a𝕍ar(𝒦, X; prewhite=prewhite)
    @test Σ ≈ Σ₀ rtol=1e-6
  end
end

kernels = (HR0(), HR1(), HR2(), HR3(), HR4(), HR4m(), HR5())

Σ₀ = [ 0.06415873470586395 -0.004015858202035743 -6.709834054283887e-5;
-0.004015858202035743 0.07800552644879759 -0.00615707811722861;
-6.709834054283887e-5 -0.00615707811722861 0.07184846516936118]

@testset "Heteroskedasticity Robust (HR) Estimators ✅" begin
  for 𝒦 in kernels
    Σ = a𝕍ar(𝒦, X)
    @test Σ ≈ Σ₀ rtol=1e-6
  end
end

@testset "Cluster Robust HC ✅" begin
  cl = repeat(1:5, inner=20)
  𝒦 = CR0(cl)
  Σ = a𝕍ar(𝒦, X)
  ## sandwich package uses HC0 but always scales by G/(G-1)
  ## below is the result of
  ## v = vcovCL(lm(X~1), cl, type="HC0")
  Σ₀ =  [0.0013477493837805246 0.0001411950613289987 4.6345925014758175e-5;
  0.0001411950613289987 0.0004985361461159058 -0.00039126414097571385;
  4.6345925014758175e-5 -0.00039126414097571385 0.00033308110226548546]
  @test Σ*5/(5-1) ≈ Σ₀*100 rtol = 1e-8
  ## Since a𝕍ar is scaled by (G/n^2), this is equivalent to  dividing by (1/G) to get the
  ## standard error and then multiply by G/(G-1) to apply the correction.
end

@testset "Driscoll & Kraay ✅" begin
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
  𝒦 = CovarianceMatrices.DriscollKraay(Bartlett(bw), tis=df.Year, iis=df.Firm)
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

@testset "Smoothed HAC ✅" begin
    X = randn(StableRNG(12322), 3700000, 3)
    k = BartlettSmoother(3)
    Σₛ = a𝕍ar(k, X; demean = true)
    Σₕ = a𝕍ar(Parzen(3), X; demean = true)
    @test Σₛ ≈ Σₕ rtol = 1e-3
    k = TruncatedSmoother(3)
    Σₛ = a𝕍ar(k, X; demean = true)
    Σₕ = a𝕍ar(Bartlett(3), X; demean = true)
    @test Σₛ ≈ Σₕ rtol = 1e-3
end

@testset "Type Promotion ✅" begin
    X = randn(StableRNG(123), 100, 3)
    Z = mapreduce(x->x.<=0, hcat, eachcol(X))
    aVar(HC0(), Z)
    aVar(Bartlett(2), Z)
    aVar(Bartlett{NeweyWest}(), Z)
end

@testset "Type Stability ✅" begin
    X = randn(StableRNG(123), Float32, 100, 3)
    @test eltype(aVar(HC0(), X)) == Float32
    @test eltype(aVar(Parzen(2), X)) == Float32
    @test eltype(aVar(Parzen{NeweyWest}(), X)) == Float32
end

end # Core Functionality Tests

@testset "GLM Integration Tests" begin

## Test GLM Interface
andrews_kernels = [:Truncated, :Parzen, :TukeyHanning, :QuadraticSpectral, :Bartlett]
neweywest_kernels = [:Parzen, :QuadraticSpectral, :Bartlett]

reg = JSON.parse(read(joinpath(datadir, "testdata/regression.json"), String))
wreg = JSON.parse(read(joinpath(datadir, "testdata/wregression.json"), String))
df = CSV.File(joinpath(datadir, "testdata/ols_df.csv")) |> DataFrame

function fopt!(u; weighted=false)
  global da = Dict{String, Any}()
  global dn = Dict{String, Any}()
  global hr_glm = Dict{String, Any}()
  global hr_lm = Dict{String, Any}()
  for pre in (:false, :true)
    da["bwtype"] = "auto"
    da["prewhite"] = pre == :true ? true : false
    dn["bwtype"] = "auto"
    dn["prewhite"] = pre == :true ? true : false

    for k in andrews_kernels
      eval(quote
        ols = glm(@formula(y~x1+x2+x3), $df, Normal(), IdentityLink(), wts=$weighted ? $(df).w : Float64[])
        𝒦 = ($k){Andrews}()
        tmp = vcov(𝒦, ols; prewhite=$pre, dofadjust=false)
        da[String($k)] = Dict{String, Any}("bw" => CM.bandwidth(𝒦), "V" => tmp)
          end)
    end

    for k in neweywest_kernels
      eval(quote
        𝒦 = ($k){NeweyWest}()
        ## To get the same results of R, the weights given to the intercept should be 0
        tmp = vcov(𝒦, ols; prewhite=$pre, dofadjust=false)
        dn[String($k)] = Dict{String, Any}("bw" => CM.bandwidth(𝒦), "V" => tmp)
          end)
    end
    push!(u, Dict("andrews" => da, "neweywest" => dn))
    da = Dict{String, Any}()
    dn = Dict{String, Any}()

  end

  for k in Symbol.(("HC".*[string.(0:4); "4m"; "5"]))
    eval(quote
          𝒦 = ($k)()
          ## To get the same results of R, the weights given to the intercept should be 0
          tmp = vcov(𝒦, ols; dofadjust=false)
          hr_glm[String($k)] = Dict{String, Any}("V" => tmp)
          end)
    eval(quote
          𝒦 = ($k)()
          ols = lm(@formula(y~x1+x2+x3), $df, wts=$weighted ? $(df).w : Float64[])
          ## To get the same results of R, the weights given to the intercept should be 0
          tmp = vcov(𝒦, ols; dofadjust=false)
          hr_lm[String($k)] = Dict{String, Any}("V" => tmp)
          end)

  end
  push!(u, Dict("hr_lm" => hr_lm, "hr_glm" => hr_glm))
  return u
end

function ffix!(u; weighted=false)
  global da = Dict{String, Any}()
  global dn = Dict{String, Any}()
  for pre in (:false, :true)
    da["bwtype"] = "fixed"
    da["prewhite"] = pre == :true ? true : false
    dn["bwtype"] = "fixed"
    dn["prewhite"] = pre == :true ? true : false
    for k in andrews_kernels
      eval(quote
            ols = glm(@formula(y~x1+x2+x3), $df, Normal(), IdentityLink(), wts=$weighted ? $(df).w : Float64[])
            𝒦 = ($k)(3)
            tmp = vcov(𝒦, ols; prewhite=$pre, dofadjust=false)
            da[String($k)] = Dict{String, Any}("bw" => CM.bandwidth(𝒦), "V" => tmp)
          end)
    end
    for k in neweywest_kernels
      eval(quote
            𝒦 = ($k)(3)
            ## To get the same results of R, the weights given to the intercept should be 0
            tmp = vcov(𝒦, ols; prewhite=$pre, dofadjust=false)
            dn[String($k)] = Dict{String, Any}("bw" => CM.bandwidth(𝒦), "V" => tmp)
          end)
    end
    push!(u, Dict("andrews" => da, "neweywest" => dn))
    da = Dict{String, Any}()
    dn = Dict{String, Any}()
  end
end

u = Any[]
fopt!(u)
ffix!(u)

@testset "Linear Model HAC ✅" begin
  for j in 1:2, h in ("andrews",), k in ("Truncated", "Bartlett", "Tukey-Hanning", "Quadratic Spectral")
    @test hcat(reg[j][h][k]["V"]...) ≈ u[j][h][k]["V"]
    @test reg[j][h][k]["bw"] ≈ u[j][h][k]["bw"]
  end

  for j in 1:2, h in ("neweywest",), k in ("Bartlett", "Quadratic Spectral")
    @test hcat(reg[j][h][k]["V"]...) ≈ u[j][h][k]["V"]
    @test reg[j][h][k]["bw"] ≈ u[j][h][k]["bw"]
  end

  for j in 3:4, h in ("andrews",), k in ("Truncated", "Bartlett", "Tukey-Hanning", "Quadratic Spectral")
    @test hcat(reg[j][h][k]["V"]...) ≈ u[j+1][h][k]["V"]
    @test reg[j][h][k]["bw"] ≈ u[j+1][h][k]["bw"]
  end

  for j in 3:4, h in ("neweywest",), k in ("Bartlett", "Quadratic Spectral")
    @test hcat(reg[j][h][k]["V"]...) ≈ u[j+1][h][k]["V"]
    @test reg[j][h][k]["bw"] ≈ u[j+1][h][k]["bw"]
  end
end

uw = Any[]
fopt!(uw; weighted=true)
ffix!(uw; weighted=true)

@testset "Linear Model HAC (Weighted) ✅" begin
  for j in 1:2, h in ("andrews",), k in ("Truncated", "Bartlett", "Tukey-Hanning", "Quadratic Spectral")
    @test hcat(wreg[j][h][k]["V"]...) ≈ uw[j][h][k]["V"]
    @test wreg[j][h][k]["bw"] ≈ uw[j][h][k]["bw"]
  end

  for j in 1:2, h in ("neweywest",), k in ("Bartlett", "Quadratic Spectral")
    @test hcat(wreg[j][h][k]["V"]...) ≈ uw[j][h][k]["V"]
    @test wreg[j][h][k]["bw"] ≈ uw[j][h][k]["bw"]
  end

  for j in 3:4, h in ("andrews",), k in ("Truncated", "Bartlett", "Tukey-Hanning", "Quadratic Spectral")
    @test hcat(wreg[j][h][k]["V"]...) ≈ uw[j+1][h][k]["V"]
    @test wreg[j][h][k]["bw"] ≈ uw[j+1][h][k]["bw"]
  end

  for j in 3:4, h in ("neweywest",), k in ("Bartlett", "Quadratic Spectral")
    @test hcat(wreg[j][h][k]["V"]...) ≈ uw[j+1][h][k]["V"]
    @test wreg[j][h][k]["bw"] ≈ uw[j+1][h][k]["bw"]
  end
end

@testset "Linear Model HC ✅" begin
for k in ("HR".*[string.(1:4); "4m"; "5"])
  @test hcat(reg[5]["hr"][k]["V"]...) ≈ u[3]["hr_glm"][k]["V"]
  @test hcat(reg[5]["hr"][k]["V"]...) ≈ u[3]["hr_lm"][k]["V"]
  @test hcat(wreg[5]["hr"][k]["V"]...) ≈ uw[3]["hr_glm"][k]["V"]
  @test hcat(wreg[5]["hr"][k]["V"]...) ≈ uw[3]["hr_lm"][k]["V"]
end
end

df = CSV.File(joinpath(datadir, "testdata/ols_df.csv")) |> DataFrame
df.w = rand(StableRNG(123), size(df,1))

@testset "Deficient Rank ✅" begin
  df.z = df.x1+df.x2
  lm1 = lm(@formula(y~x1+x2+x3+z), df, wts=df.w)
  lmm = lm(@formula(y~x2+x3+z), df, wts=df.w)
  V1 = vcov(HC1(), lm1)
  V2 = vcov(HC1(), lmm)
  idxr = isempty.(map(i->findall(all(isnan.(i))), eachrow(V1)))
  idxc = isempty.(map(i->findall(all(isnan.(i))), eachcol(V1)))
end

@testset "Linear Model Cluster Robust ✅" begin
  df = CSV.read(joinpath(datadir,"testdata/PetersenCl.csv"), DataFrame)
  m = lm(@formula(y~x), df)
  V0 = vcov(CR0(df.firm), m)
  V1 = vcov(CR1(df.firm), m)
  V2 = vcov(CR2(df.firm), m)
  V3 = vcov(CR3(df.firm), m)
  R0 = [ 0.0044808245285903594 -6.4592772035200005e-05;
        -6.4592772035200005e-05 0.0025542965590391016]
  R1 = [ 0.0044907024570195212  -6.4735166091279167e-05;
        -6.4735166091279154e-05  0.002559927477731868]
  R2 = [ 0.0044944872570531966  -6.5929118692432861e-05;
        -6.5929118692432861e-05  0.0025682360417855431]
  R3 = [ 0.0045082022937877244   -6.7280836115793117e-05;
        -6.7280836115793117e-05   0.0025822624320339092]
  @test V0 ≈ R0 rtol=1e-6
  @test V1 ≈ R1 rtol=1e-6
  @test V2 ≈ R2 rtol=1e-4
  @test V3 ≈ R3 rtol=1e-6

  df.w = rand(StableRNG(123), size(df,1))
  m = lm(@formula(y~x), df, wts=df.w)
  V0 = vcov(CR0(df.firm), m)
  V1 = vcov(CR1(df.firm), m)
  V2 = vcov(CR2(df.firm), m)
  V3 = vcov(CR3(df.firm), m)
  R0 = [ 0.0048945800009019078 -9.5217611657051797e-05;
        -9.5217611657051783e-05 0.0029666072693556918]
  R1 = [ 0.0049053700487226215  -9.5427517835511321e-05;
        -9.5427517835511335e-05  0.0029731471224780704]
  R2 = [ 0.0049055911851245753 -9.6888552025056022e-05;
        -9.6888552025055995e-05 0.0029835249179155422]
  R3 = [ 0.0049264936435635398 -9.8773895065056182e-05;
        -9.8773895065056168e-05 0.0030065617267332799]
  @test V0 ≈ R0
  @test V1 ≈ R1
  @test V2 ≈ R2 atol=1e-5
  @test V3 ≈ R3 atol=1e-5
end

@testset "GLM HAC ✅" begin
    clotting = DataFrame(
        u    = log.([5,10,15,20,30,40,60,80,100]),
        lot1 = [118,58,42,35,27,25,21,19,18],
        lot2 = [69,35,26,21,18,16,13,12,12],
        w    = 9.0*[1/8, 1/9, 1/25, 1/6, 1/14, 1/25, 1/15, 1/13, 0.3022039]
    )

    GAMMA = glm(@formula(lot1~u), clotting, Gamma(),InverseLink(), wts = convert(Array, clotting[!, :w]))
    V = vcov(Parzen{Andrews}(), GAMMA)
    Vp = [5.48898e-7 -2.60409e-7; -2.60409e-7 1.4226e-7]
    @test V ≈ Vp atol = 1e-08

    GAMMA = glm(@formula(lot1~u), clotting, Gamma(),InverseLink())
    k = Parzen{Andrews}()
    V = vcov(k, GAMMA)
    Vp = [5.81672e-7 -2.24162e-7; -2.24162e-7 1.09657e-7]
    @test V ≈ Vp atol = 1e-08
end

@testset "GLM HC ✅" begin
    # A Gamma example, from McCullagh & Nelder (1989, pp. 300-2)
    clotting = DataFrame(
        u    = log.([5,10,15,20,30,40,60,80,100]),
        lot1 = [118,58,42,35,27,25,21,19,18],
        lot2 = [69,35,26,21,18,16,13,12,12],
        w    = 9.0*[1/8, 1/9, 1/25, 1/6, 1/14, 1/25, 1/15, 1/13, 0.3022039]
    )

    ## Unweighted OLS though GLM interface
    OLS = fit(GeneralizedLinearModel, @formula(lot1~u),clotting, Normal(), IdentityLink())
    mf = ModelFrame(@formula(lot1~u),clotting)
    X = ModelMatrix(mf).m
    y = clotting[!, :lot1]
    GL  = fit(GeneralizedLinearModel, X,y, Normal(), IdentityLink())
    LM  = lm(X,y)

    S0 = vcov(HC0(),OLS)
    S1 = vcov(HC1(),OLS)
    S2 = vcov(HC2(),OLS)
    S3 = vcov(HC3(),OLS)
    S4 = vcov(HC4(),OLS)
    S4m = vcov(HC4m(),OLS)
    S5 = vcov(HC5(),OLS)

    St0 = [720.621306411 -190.064512543; -190.064512543 51.163333742]
    St1 = [926.513108242 -244.368658984; -244.368658984 65.781429097]
    St2 = [1300.895673284 -343.330672699; -343.330672699  91.997186033]
    St3 = [2384.50393347 -628.97499232; -628.97499232 167.78976878]
    St4 = [2538.74635384 -667.95972319; -667.95972319 177.26308957]
    St4m = [3221.09520169 -849.64802585; -849.64802585 226.17046981]
    St5 = [1334.670541439 -351.751377823; -351.751377823  93.949230276]


    @test S0 ≈ St0
    @test S1 ≈ St1
    @test S2 ≈ St2
    @test S3 ≈ St3
    @test S4 ≈ St4
    @test S4m ≈ St4m
    @test S5 ≈ St5

    S0 = vcov(HC0(), GL)
    S1 = vcov(HC1(), GL)
    S2 = vcov(HC2(), GL)
    S3 = vcov(HC3(), GL)
    S4 = vcov(HC4(), GL)
    S4m = vcov(HC4m(), GL)
    S5 = vcov(HC5(), GL)

    @test S0 ≈ St0
    @test S1 ≈ St1
    @test S2 ≈ St2
    @test S3 ≈ St3
    @test S4 ≈ St4
    @test S4m ≈ St4m
    @test S5 ≈ St5

    ## Weighted OLS though GLM interface
    wOLS = fit(GeneralizedLinearModel, @formula(lot1~u), clotting, Normal(),
               IdentityLink(), wts = Vector{Float64}(clotting[!, :w]))

    wts = Vector{Float64}(clotting[!, :w])
    X = [fill(1,size(clotting[!, :u])) clotting[!, :u]]
    y = clotting[!, :lot1]
    wLM = lm(X, y)
    wGL = fit(GeneralizedLinearModel, X, y, Normal(), IdentityLink(), wts = wts)

    S0 = vcov(HC0(),wOLS)
    S1 = vcov(HC1(),wOLS)
    S2 = vcov(HC2(),wOLS)
    S3 = vcov(HC3(),wOLS)
    S4 = vcov(HC4(),wOLS)
    S4m= vcov(HC4m(),wOLS)
    S5 = vcov(HC5(),wOLS)

    St0 = [717.736178076 -178.404274981; -178.404274981   45.822730697]
    St1 = [922.803657527 -229.376924975; -229.376924975 58.914939468]
    St2 = [1412.940497584 -361.329969345; -361.329969345 95.912520696]
    St3 = [2869.53068690 -756.29761027; -756.29761027 208.23437869]
    St4 = [3969.9130263 -1131.3577578; -1131.3577578 342.2858663]
    St4m= [4111.62611908 -1103.17362711; -1103.17362711   310.19430896]
    St5 = [1597.40932634 -420.66907485; -420.66907485 115.99180777]

    @test S0  ≈ St0
    @test S1  ≈ St1
    @test S2  ≈ St2
    @test S3  ≈ St3
    @test S4  ≈ St4
    @test S4m ≈ St4m
    @test S5  ≈ St5

    ## Unweighted GLM - Gamma
    GAMMA = glm(@formula(lot1~u), clotting, Gamma(),InverseLink())

    S0 = vcov(HC0(),GAMMA)
    S1 = vcov(HC1(),GAMMA)
    S2 = vcov(HC2(),GAMMA)
    S3 = vcov(HC3(),GAMMA)
    S4 = vcov(HC4(),GAMMA)
    S4m = vcov(HC4m(),GAMMA)
    S5 = vcov(HC5(),GAMMA)

    St0 = [4.504287921232951e-07 -1.700020601541489e-07;
           -1.700020601541490e-07  8.203697048568913e-08]

    St1 = [5.791227327299548e-07 -2.185740773410504e-07;
       -2.185740773410510e-07  1.054761049101728e-07]

    St2 = [3.192633083111232e-06 -9.942484630848573e-07;
           -9.942484630848578e-07  3.329973305723091e-07]

    St3 = [2.982697811926944e-05 -8.948137019946751e-06;
           -8.948137019946738e-06  2.712024459305714e-06]

    St4 = [0.002840158946368653 -0.0008474436578800430;
           -0.000847443657880045  0.0002528819761961959]

    St4m= [9.2891282926e-05  -2.7759505159e-05;
           -2.7759505159e-05  8.3203461732e-06]

    St5 = [2.9781374021e-05  -8.9232514073e-06
           -8.9232514073e-06  2.6952175350e-06]

    @test S0  ≈ St0 atol = 1e-08
    @test S1  ≈ St1 atol = 1e-08
    @test S2  ≈ St2 atol = 1e-08
    @test S3  ≈ St3 atol = 1e-06
    @test S4  ≈ St4 atol = 1e-05
    @test S4m ≈ St4m atol = 1e-06
    @test S5  ≈ St5 atol = 1e-06

    ## Weighted Gamma
    GAMMA = glm(@formula(lot1~u), clotting, Gamma(),InverseLink(), wts = convert(Array, clotting[!, :w]))

    S0 = vcov(HC0(),GAMMA)
    S1 = vcov(HC1(),GAMMA)
    S2 = vcov(HC2(),GAMMA)
    S3 = vcov(HC3(),GAMMA)
    S4 = vcov(HC4(),GAMMA)
    S4m = vcov(HC4m(),GAMMA)
    S5 = vcov(HC5(),GAMMA)

    St0 = [4.015104e-07 -1.615094e-07;
           -1.615094e-07  8.378363e-08]

    St1 = [5.162277e-07 -2.076549e-07;
           -2.076549e-07  1.077218e-07]

    St2 = [2.720127e-06 -8.490977e-07;
           -8.490977e-07  2.963563e-07]

    St3 = [2.638128e-05 -7.639883e-06;
           -7.639883e-06  2.259590e-06]

    St4 = [0.0029025754 -0.0008275858;
           -0.0008275858  0.0002360053]

    St4m = [8.493064e-05 -2.436180e-05;
            -2.436180e-05  7.042101e-06]

    St5 = [2.6206554518e-05 -7.5421496876e-06
          -7.5421496876e-06  2.2017813312e-06]

    @test S0  ≈ St0 atol = 1e-08
    @test S1  ≈ St1 atol = 1e-08
    @test S2  ≈ St2 atol = 1e-08
    @test S3  ≈ St3 atol = 1e-07
    @test S4  ≈ St4 atol = 1e-05
    @test S4m ≈ St4m atol = 1e-07
    @test S5  ≈ St5 atol = 1e-07
end

end # GLM Integration Tests