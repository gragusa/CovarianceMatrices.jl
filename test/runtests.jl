using CovarianceMatrices
using Test
using TableReader
using LinearAlgebra
using Statistics
using Random
using GLM
using DataFrames
using JSON
using StatsBase
using CategoricalArrays

const CM = CovarianceMatrices
const andrews_kernels = [:TruncatedKernel, :ParzenKernel, :TukeyHanningKernel, :QuadraticSpectralKernel, :BartlettKernel]
const neweywest_kernels = [:ParzenKernel, :QuadraticSpectralKernel, :BartlettKernel]

datapath = joinpath(@__DIR__)

@testset "BW Optimal......................................." begin
    Random.seed!(1)
    df = DataFrame(y=randn(20), x=randn(20))
    lm1 = lm(@formula(y~x), df)
    k = BartlettKernel{NeweyWest}()
    V = vcov(k, lm1; prewhite=true)
    bw = optimal_bandwidth(BartlettKernel{NeweyWest}(), lm1, prewhite=true)
    V2 = vcov(BartlettKernel(bw), lm1; prewhite=true)
    @test bw==first(k.bw)
    @test V≈V2

    k = BartlettKernel{NeweyWest}()
    V = vcov(k, lm1; prewhite=false)
    bw = optimal_bandwidth(BartlettKernel{NeweyWest}(), lm1, prewhite=false)
    V2 = vcov(BartlettKernel(bw), lm1; prewhite=false)
    @test bw==first(k.bw)
    @test V≈V2

    k = BartlettKernel{Andrews}()
    V = vcov(k, lm1; prewhite=false)
    bw = optimal_bandwidth(BartlettKernel{Andrews}(), lm1, prewhite=false)
    V2 = vcov(BartlettKernel(bw), lm1; prewhite=false)
    @test bw==first(k.bw)
    @test V≈V2

    k = BartlettKernel{Andrews}()
    V = vcov(k, lm1; prewhite=true)
    bw = optimal_bandwidth(BartlettKernel{Andrews}(), lm1, prewhite=true)
    V2 = vcov(BartlettKernel(bw), lm1; prewhite=true)
    @test bw==first(k.bw)
    @test V≈V2
end

@testset "HAC - Asymptotic Covariance (Fixed).............." begin
    Random.seed!(1)
    X = randn(20,2)
    V = lrvar(TruncatedKernel(2), X)./20
    Vr = [0.023797696862680198 -0.006551463062455593; -0.006551463062455593 0.05722230754875061]
    @test V ≈ Vr
    V = lrvar(TruncatedKernel(2), X, prewhite=true)./20
    Vr = [0.02306195412692154 -0.002321811843500563; -0.0023218118435005637 0.0752916863840604]
    @test V ≈ Vr
    V = lrvar(BartlettKernel(2), X, prewhite=false)./20
    Vr = [0.035714374800934555 0.000210291773413605; 0.000210291773413605 0.05629043339567539]
    @test V ≈ Vr
    V = lrvar(BartlettKernel(2), X, prewhite=true)./20
    Vr = [0.026762130596397222 0.0023844392352094616; 0.0023844392352094625 0.07539513026131048]
    @test V ≈ Vr
    V = lrvar(ParzenKernel(2), X, prewhite=false)./20
    Vr = [0.04082918144700371 0.00029567273165852334; 0.00029567273165852334 0.05178620682214104]
    @test V ≈ Vr
    V = lrvar(ParzenKernel(2), X, prewhite=true)./20
    Vr = [0.027407120851688418 0.0018031540982724296; 0.0018031540982724307 0.0702092071421895]
    @test V ≈ Vr
end
@testset "HAC - Asymptotic Covariance (Andrews)............" begin
    Random.seed!(1)
    X = randn(20,2)
    V = lrvar(TruncatedKernel{Andrews}(), X)./20
    Vr = [0.025484761508796222 3.9529856923768415e-5; 3.9529856923768415e-5 0.0652988865427441]
    @test V ≈ Vr
    V = lrvar(TruncatedKernel{Andrews}(), X, prewhite=true)./20
    Vr = [0.028052111106979607 0.001221868961335397; 0.001221868961335396 0.06502328402306855]
    @test V ≈ Vr
    V = lrvar(BartlettKernel{Andrews}(), X, prewhite=false)./20
    Vr = [0.036560369091632905 0.00022441387190291295; 0.00022441387190291295 0.0555454296782523]
    @test V ≈ Vr
    V = lrvar(BartlettKernel{Andrews}(), X, prewhite=true)./20
    Vr = [0.027270747095805972 0.001926058338627125; 0.0019260583386271257 0.07130569473664604]
    @test V ≈ Vr
    V = lrvar(ParzenKernel{Andrews}(), X, prewhite=false)./20
    Vr = [0.03100269416916425 -0.0017562565506448816; -0.0017562565506448816 0.05796097557032457]
    @test V ≈ Vr
    V = lrvar(ParzenKernel{Andrews}(), X, prewhite=true)./20
    Vr = [0.025984224823188636 0.0017705442482664063; 0.001770544248266405 0.07693448665125002]
    @test V ≈ Vr
end
@testset "HAC - Asymptotic Covariance (Newey).............." begin
end
@testset "HC  - Asymptotic Covariance......................" begin
    Random.seed!(1)
    X = randn(20,2)
    v = X .- mean(X, dims = 1)
    V = lrvar(HC0(), X)
    @test V ≈ v'*v/20
    V = lrvar(HC1(), X)
    @test V ≈ v'*v/20
    V = lrvar(HC2(), X)
    @test V ≈ v'*v/20
    V = lrvar(HC3(), X)
    @test V ≈ v'*v/20
    V = lrvar(HC0(), X, demean=false)
    @test V ≈ X'*X/20
end
@testset "CRHC  - Asymptotic Covariance...................." begin
    Random.seed!(1)
    X = randn(100,2)
    f = repeat(1:20, inner=5)
    V = lrvar(CRHC0(f), X)/100
    Vr = [0.010003084285822686 0.002579249460680671; 0.002579249460680671 0.014440606823274103]
    @test V ≈ Vr*(19/20)
    V = lrvar(CRHC1(f), X)/100
    @test V ≈ Vr*(19/20)
    V = lrvar(CRHC2(f), X)/100
    @test V ≈ Vr*(19/20)
    V = lrvar(CRHC3(f), X)/100
    @test V ≈ Vr*(19/20)
end

@testset "HAC OLS VCOV....................................." begin
    reg = JSON.parse(read("testdata/regression.json", String))
    df = TableReader.readcsv("testdata/ols_df.csv")
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
                     tmp = vcovmatrix(($k){Andrews}(), ols; prewhite=$pre, dof_adjustment = false)
                     da[String($k)] = Dict{String, Any}("bw" => tmp.K.bw, "V" => tmp.V)
                     if Symbol($k) in neweywest_kernels
                     tmp = vcovmatrix(($k){Andrews}(), ols; prewhite=$pre, dof_adjustment = false)
                     dn[String($k)] = Dict{String, Any}("bw" => tmp.K.bw, "V" => tmp.V)
                     end
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
                     tmp = vcovmatrix(($k)(1.5), ols; prewhite=$pre, dof_adjustment=false)
                     da[String($k)] = Dict{String, Any}("bw" => tmp.K.bw, "V" => tmp.V)
                     if Symbol($k) in neweywest_kernels
                         tmp = vcovmatrix(($k)(1.5), ols; prewhite=$pre, dof_adjustment=false)
                         dn[String($k)] = Dict{String, Any}("bw" => tmp.K.bw, "V" => tmp.V)
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
    ffix!(u)
    for j in 1:4, h in ("andrews",), k in ("Truncated", "Bartlett", "Tukey-Hanning", "Quadratic Spectral")
        @test hcat(reg[j][h][k]["V"]...) ≈ u[j][h][k]["V"]
        @test reg[j][h][k]["bw"] ≈ u[j][h][k]["bw"]
    end

end

@testset "HAC - GLM........................................" begin
    clotting = DataFrame(
        u    = log.([5,10,15,20,30,40,60,80,100]),
        lot1 = [118,58,42,35,27,25,21,19,18],
        lot2 = [69,35,26,21,18,16,13,12,12],
        w    = 9.0*[1/8, 1/9, 1/25, 1/6, 1/14, 1/25, 1/15, 1/13, 0.3022039]
    )

    GAMMA = glm(@formula(lot1~u), clotting, Gamma(),InverseLink(), wts = convert(Array, clotting[!, :w]))
    V = vcov(ParzenKernel{Andrews}(), GAMMA)
    Vp = [5.48898e-7 -2.60409e-7; -2.60409e-7 1.4226e-7]
    @test V ≈ Vp atol = 1e-08

    GAMMA = glm(@formula(lot1~u), clotting, Gamma(),InverseLink())
    V = vcov(ParzenKernel{Andrews}(), GAMMA)
    Vp = [5.81672e-7 -2.24162e-7; -2.24162e-7 1.09657e-7]
    @test V ≈ Vp atol = 1e-08
end

@testset "HC..............................................." begin
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
    wGL = fit(GeneralizedLinearModel, X, y, Normal(),
              IdentityLink(), wts = wts)


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

@testset "CRHC............................................." begin
    df = TableReader.readcsv("testdata/wols_test.csv")
    df_sorted = sort(df, [:X1])

    St1 = [.0374668, .0497666, .0472636, .0437952, .0513613, .0435369]

    OLS = fit(GeneralizedLinearModel, @formula(Y~X1+X2+X3+X4+X5), df, Normal(), IdentityLink())
    cl = convert(Array, df[!, :cl])
    k0 = CRHC0(cl)
    k1 = CRHC1(cl)
    k2 = CRHC2(cl)
    k3 = CRHC3(cl)

    V0 = vcov(k0, OLS)
    V1 = vcov(k1, OLS)
    V2 = vcov(k2, OLS)
    V3 = vcov(k3, OLS)

    @test V1 ≈ [0.00140376 0.000215526 -5.99768e-5 0.000296271 0.000460622 -0.000139741;
                0.000215526 0.00247671 -0.000270429 0.000218622 0.000610127 7.23345e-5;
                -5.99768e-5 -0.000270429 0.00223385 -0.000145166 -0.00018859 -0.000903561;
                0.000296271 0.000218622 -0.000145166 0.00191802 -0.000444364 -0.000420563;
                0.000460622 0.000610127 -0.00018859 -0.000444364 0.00263798 0.000736363;
                -0.000139741 7.23345e-5 -0.000903561 -0.000420563 0.000736363 0.00189546] atol = 1e-08

    ## Note sandwich in R has HC3 without G/(G-1) and CRHC2 is problematic

    @test sqrt.(diag(vcov(k0, OLS))) == sqrt.(diag(V0))
    @test sqrt.(diag(vcov(k0, OLS))) == stderror(k0, OLS)
    OLS_sorted = fit(GeneralizedLinearModel, @formula(Y~X1+X2+X3+X4+X5), df_sorted, Normal(), IdentityLink())
    cl = convert(Array, df_sorted[!, :cl])
    k0 = CRHC0(cl)
    k1 = CRHC1(cl)
    k2 = CRHC2(cl)
    k3 = CRHC3(cl)
    V0s = vcov(k0, OLS_sorted)
    V1s = vcov(k1, OLS_sorted)
    V2s = vcov(k2, OLS_sorted)
    V3s = vcov(k3, OLS_sorted)

    @test V0s ≈ V0
    @test V1s ≈ V1
    @test V2s ≈ V2 atol=1e-04
    @test V3s ≈ V3

    wOLS = fit(GeneralizedLinearModel, @formula(Y~X1+X2+X3+X4+X5), df,
               Normal(), IdentityLink(), wts = convert(Array{Float64}, df[!, :w]))

    cl = convert(Array, df[!, :cl])
    k0 = CRHC0(cl)
    k1 = CRHC1(cl)
    k2 = CRHC2(cl)
    k3 = CRHC3(cl)

    V0 = vcov(k0, wOLS)
    V1 = vcov(k1, wOLS)
    V2 = vcov(k2, wOLS)
    V3 = vcov(k3, wOLS)

    @test V1 ≈ [0.00183525 0.000137208 -0.00038971 0.000389943 0.000619903 0.00019496;
                0.000137208 0.00242781 -0.000272316 0.000462353 2.99597e-5 0.000133303;
                -0.00038971 -0.000272316 0.00273479 -0.000113765 -7.26396e-5 -0.000998524;
                0.000389943 0.000462353 -0.000113765 0.00171538 -0.00067357 -0.000416268;
                0.000619903 2.99597e-5 -7.26396e-5 -0.00067357 0.00225446 0.00106796;
                0.00019496 0.000133303 -0.000998524 -0.000416268 0.00106796 0.00226444] atol = 1e-07

    innovation = TableReader.readcsv("testdata/InstInnovation.csv")

    innovation[!, :capemp] = log.(innovation[!, :capital]./innovation[!, :employment])
    innovation[!, :lsales] = log.(innovation[!, :sales])
    innovation[!, :year] = categorical(innovation[!, :year])
    innovation[!, :industry] = categorical(innovation[!, :industry])
    #innovation[:company] = categorical(innovation[:company])
    pois = glm(@formula(cites ~ institutions + capemp + lsales + industry + year), innovation, Poisson(), LogLink())
    Vt = [0.904094640946072,
          0.00240638781048165,
          0.135953255431155,
          0.0415234048672968]

    @test sqrt.(diag(vcov(CRHC0(innovation[!, :company]), pois))[1:4]) ≈ Vt atol = 1e-5
end


# @testset "CovarianceMatrices Methods......................." begin
#     df = TableReader.readcsv("testdata/wols_test.csv")
#     df_sorted = sort!(copy(df), :cl)
#     cl = convert(Array, df[!, :cl])
#     wOLS = fit(GeneralizedLinearModel, @formula(Y~X1+X2+X3+X4+X5), df,
#                Normal(), IdentityLink(), wts = convert(Array{Float64}, df[!, :w]))

#     V0 = vcov(HC0(), wOLS)
#     V1 = vcov(ParzenKernel(), wOLS)
#     V2 = vcov(CRHC0(cl), wOLS)

#     V0s = vcov(HC0(), wOLS, CovarianceMatrix, SVD)
#     V1s = vcov(ParzenKernel(), wOLS, CovarianceMatrix, SVD)
#     V2s = vcov(CRHC0(cl), wOLS, CovarianceMatrix, SVD)

#     V0s2 = vcov(HC0(), wOLS,  SVD)
#     V1s2 = vcov(ParzenKernel(), wOLS,  SVD)
#     V2s2 = vcov(CRHC0(cl), wOLS,  SVD)

#     V0m = vcov(HC0(), wOLS, Matrix)
#     V1m = vcov(ParzenKernel(), wOLS, Matrix)
#     V2m = vcov(CRHC0(cl), wOLS, Matrix)

#     V0c = vcov(HC0(), wOLS, CovarianceMatrix, Cholesky)
#     V1c = vcov(ParzenKernel(), wOLS, CovarianceMatrix, Cholesky)
#     V2c = vcov(CRHC0(cl), wOLS, CovarianceMatrix, Cholesky)

#     V0c2 = vcov(HC0(), wOLS, Cholesky)
#     V1c2 = vcov(ParzenKernel(), wOLS, Cholesky)
#     V2c2 = vcov(CRHC0(cl), wOLS, Cholesky)

#     @test V0 == V0m
#     @test V1 == V1m
#     @test V2 == V2m

#     @test V0c.F == cholesky(Symmetric(V0m))
#     @test V1c.F == cholesky(Symmetric(V1m))
#     @test V2c.F == cholesky(Symmetric(V2m))

#     @test V0c2.F == cholesky(Symmetric(V0m))
#     @test V1c2.F == cholesky(Symmetric(V1m))
#     @test V2c2.F == cholesky(Symmetric(V2m))

#     @test V0s.F == svd(V0m)
#     @test V1s.F == svd(V1m)
#     @test V2s.F == svd(V2m)

#     @test V0s2.F == svd(V0m)
#     @test V1s2.F == svd(V1m)
#     @test V2s2.F == svd(V2m)

#     @test inv(V2s2) ≈ inv(Matrix(V2s2))
#     @test inv(V2c2) ≈ inv(Matrix(V2c2))

# end

@testset "Various.........................................." begin
    Random.seed!(1)
    Z = randn(10, 5)
    @test lrvar(HC1(), Z, demean = true) ≈ cov(StatsBase.SimpleCovariance(), Z)
    @test lrvar(HC1(), Z, demean = true) ≈ lrvar(HC1(), Z .- mean(Z, dims=1), demean = false)
    @test lrvar(HC1(), Z, demean = false) ≈ Z'Z/size(Z,1)
    @test lrvarmatrix(HC1(), Z; demean = true) ≈
        cov(StatsBase.SimpleCovariance(), Z)
    @test lrvarmatrix(HC1(), Z; demean = true) ≈
        lrvar(HC1(), Z .- mean(Z, dims=1), demean = false)
    ## Need testing for CRHC
end





# @testset "NeweyWest Optimal BW............................." begin
#     Random.seed!(9)
#     X = randn(30,5)
#     y = rand(30)
#     df = DataFrame(y = y, x1 = X[:,2], x2 = X[:,3], x3 = X[:,4], x4 = X[:,5])
#     V = [0.0014234570920295653 -0.0003203609358696263 0.0002145224188789533;
#         -0.00032036093586962626 0.002030239858282678 0.0006792335197493722;
#         0.0002145224188789535 0.0006792335197493722 0.0010400408350070508]
#     nwbw = 8.895268
#     k = ParzenKernel(NeweyWest)
#     Vj = vcov(k, lm(@formula(y~x1+x2), df))
#     @test V ≈ Vj
#     @test k.bw[1] ≈ nwbw

#     V = [0.0022881871780231766 -0.0008842029695304288 0.000442063353905825;
#     -0.0008842029695304284 0.0026011141571525057 0.0006638051071556448;
#      0.00044206335390582503 0.0006638051071556448 0.0013358590259366617]
#     nwbw = 4.23744676121782
#     k = BartlettKernel(NeweyWest)
#     Vj = vcov(k, lm(@formula(y~x1+x2), df))
#     @test V ≈ Vj
#     @test k.bw[1] ≈ nwbw

#     V = [0.0015938483461041284 -0.0004443550089371629 0.00022923187497491063;
#         -0.0004443550089371628 0.0022415134516427515 0.0007630674168323773;
#          0.0002292318749749108 0.0007630674168323773 0.0010705259313823915]
#     nwbw  = 4.41888992824362
#     k = QuadraticSpectralKernel(NeweyWest)
#     Vj = vcov(k, lm(@formula(y~x1+x2), df))
#     @test V ≈ Vj
#     @test k.bw[1] ≈ nwbw

#     V = [0.017662022128800667 -0.012810546226779051 -0.0008881499052458461 -0.007806577092014024 0.011568272552465097;
#         -0.012810546226779051 0.02892731512920205 0.005309632973967233 0.0061614622214174065 -0.007451816342478188;
#         -0.0008881499052458454 0.005309632973967234 0.009892573169329903 0.001242952114188168 -0.0027346850659789946;
#         -0.007806577092014025 0.0061614622214174065 0.0012429521141881681 0.010322298771795132 -0.013148100284155506;
#         0.011568272552465097 -0.007451816342478188 -0.0027346850659789946 -0.013148100284155504 0.028997064190718284]

#     @test  vcov(ParzenKernel(), lm(X, y)) ≈ V


# end

@testset "Covariance Matrix Methods........................" begin
    Random.seed!(9)
    X = randn(30,5)
    y = rand(30)
    df = DataFrame(y = y, x1 = X[:,2], x2 = X[:,3], x3 = X[:,4], x4 = X[:,5])
    k = TruncatedKernel(1)
    CM1 = vcovmatrix(k, lm(X,y))
    CM2 = vcovmatrix(k, lm(X,y), SVD)
    CM3 = vcovmatrix(k, lm(X,y), Cholesky)

    CMc1 = vcov(k, lm(X,y))
    CMc2 = vcov(k, lm(X,y))
    CMc3 = vcov(k, lm(X,y))

    @test CM1 ≈ CMc1
    @test CM2 ≈ CMc2
    @test CM3 ≈ CMc3


    @test CM1[1,1] == CMc1[1,1]
    @test CM1[1,1] == CMc2[1,1]

    @test size(CM1) == (5,5)
    #@test eltype(CM1) == CM.WFLOAT

    @test CM.invfact(CM3, true) ≈ inv(cholesky(Hermitian(Matrix(CM3))).L)
    @test CM.invfact(CM3, false) ≈ inv(cholesky(Hermitian(Matrix(CM3))).U)

    @test CM.invfact(CM3)'*CM.invfact(CM3) ≈ inv(CM3)


    @test CM.invfact(CM2)'*CM.invfact(CM2) ≈ inv(CM3)

    s = svd(Matrix(CM1))
    @test CM.invfact(CM2) ≈ diagm((1.0./sqrt.(s.S)))*s.Vt

    @test eigmax(CM2) ≈ eigmax(Matrix(CM2))
    @test eigmax(CM3) ≈ eigmax(Matrix(CM3))

    @test eigmin(CM2) ≈ eigmin(Matrix(CM2))
    @test eigmin(CM3) ≈ eigmin(Matrix(CM3))

    @test logdet(CM3) ≈ logdet(Matrix(CM3))
    @test logdet(CM2) ≈ logdet(Matrix(CM2))

    @test inv(CM1) ≈ pinv(Matrix(CM1))

    g = mean(X, dims = 1)
    @test CM.quadinv(g, CM1) ≈ first(g*inv(Matrix(CM1))*g')

    g = rand(5)
    @test CM.quadinv(g, CM1) ≈ first(g'*inv(Matrix(CM1))*g)

    @test Symmetric(CM1) == Symmetric(Matrix(CM1))

end

@testset "VARHAC..........................................." begin
    k = CM.VARHAC()
    Random.seed!(1)
    g = randn(100,2);
    G1 = lrvar(k, g)
    G2 = lrvar(k, g)
    k = CM.VARHAC(maxlag=2, lagstrategy=2)
    G2 = lrvar(k, g)
    k = CM.VARHAC(maxlag=3, lagstrategy=1)
    G2 = lrvar(k, g)

    k = CM.VARHAC(maxlag=3, lagstrategy=1, selectionstrategy=:bic)
    G2 = lrvar(k, g)
    k = CM.VARHAC(maxlag=3, lagstrategy=2, selectionstrategy=:bic)
    G2 = lrvar(k, g)
    k = CM.VARHAC(maxlag=3, lagstrategy=3, selectionstrategy=:bic)
    G2 = lrvar(k, g)

    k = CM.VARHAC(maxlag=3, lagstrategy=1, selectionstrategy=:eq)
    G2 = lrvar(k, g)

    k = CM.VARHAC(maxlag=3, lagstrategy=2, selectionstrategy=:eq)
    G2 = lrvar(k, g)
    k = CM.VARHAC(maxlag=3, lagstrategy=3, selectionstrategy=:eq)
    G2 = lrvar(k, g)

end
