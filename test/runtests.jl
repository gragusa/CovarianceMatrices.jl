using CovarianceMatrices
using Test
using CSV
using LinearAlgebra
using Statistics
using Random
using GLM
using DataFrames
using JSON

const CM = CovarianceMatrices

datapath = joinpath(@__DIR__)

@testset "HAC - Basic checks..............................." begin
    X = [0 0; 3 7; 4 8; 5 9]
    X_demean = [-3  -6; 0   1; 1   2; 2   3]
    cache = CM.HACCache(X, TruncatedKernel(prewhiten=true));
    CM.demean!(cache, X, Val{true})
    @test all(cache.μ .== [3 6])
    @test all(cache.X_demean .== X_demean)
    CM.fit_var!(cache)
    @test all(cache.D   .≈ [-9/5 -4; 1 2])
    @test all(cache.XX  .≈ [3/5 1; 0 0; 9/5 3])
end

@testset "HAC - Optimal Bandwidth Calculations............." begin
    X = [1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7,  .8,
         .3, .4, .5, .6, .7, .8, .9, .10, .11, .12]
    X = reshape(X, 8, 3)

    k = ParzenKernel(prewhiten=true)
    cache = CovarianceMatrices.HACCache(X, k)
    CovarianceMatrices.demean!(cache, X, Val{true})

    Xd = reshape([ -2.75,    -1.75,   -0.75,     0.25,
                   1.25,     2.25,    3.25,    -1.75,
                   -0.312500000000002,   0.6875,   1.6875,  2.6875,
                   3.6875,  -2.5125,  -3.0125, -2.9125,
                   0.0212499999999998,  0.12125,  0.22125, 0.32125,
                   0.42125, -0.37875, -0.36875, -0.35875], 8, 3)
    @test all(Xd .≈ cache.X_demean)

    CovarianceMatrices.prewhiten!(cache)

    D = reshape([-0.409214603090833,   -0.546926852567603, -0.0793279159360713,
                 7.17805137100394,    -1.09268085748226,  -0.152674132870786,
                 -56.8116543172651,    12.0360552297997,    1.52457939755027], 3,3)'

    @test all(D .≈ cache.D)

    XX = reshape([0.575048549180823,     0.487377212994233,   0.399705876807639,
                  0.312034540621058,      0.22436320443446,   0.688172853937608,
                  0.254529685703059,     -1.41377778615736,  0.0222244009125362,
                  1.45822658798243,       2.89422877505233,   -2.86976903787778,
                  0.0313806821395586,   0.0116065536680327,   -0.17700974754426,
                  0.00253436150757022,     0.1820784705594,    0.36162257961123,
                  -0.35883331133694,   0.00357849884047592, 0.00132355436565176], 7, 3)
    @test all(XX .≈ cache.XX)

    CovarianceMatrices.fit_ar!(cache)
    ρ = [-0.411560904449045,  -0.202815142161935, -0.202567957968907]
    σ⁴= [ 0.00043730337156934, 8.49515993964198,   0.0020740008652368]
    @test all(ρ .≈ cache.ρ)
    @test all(σ⁴ .≈ cache.σ⁴)

    CovarianceMatrices.fit_ar!(cache)
    if isempty(k.weights)
        for j in 1:size(X,2)
            push!(k.weights, 1.0)
        end
    end
    a1, a2 = CovarianceMatrices.getalpha!(cache, k.weights)
    @test a1 ≈ 0.1789771071933
    @test a2 ≈ 0.07861018601427
    @test CovarianceMatrices.bw_andrews(k, a1, a2, 7) ≈ 2.361704327253
    k.bw .= CovarianceMatrices.bw_andrews(k, a1, a2, 7)
    bw = k.bw[1]
    fill!(cache.V, zero(eltype(cache.V)))
    mul!(cache.V, cache.XX', cache.XX)
    V = reshape([0.707026904730492,   0.0322404268557179, 0.00367653990459955,
                 0.0322404268557179, 10.3694701634646,    1.29616765021965,
                 0.00367653990459955, 1.29616765021965,   0.162019118007504], 3, 3)*2

    @test all(V ≈ cache.V)
    triu!(cache.V)
    @test CovarianceMatrices.kernel(k, 1/bw) ≈  0.379763254768776
    @test CovarianceMatrices.kernel(k, 2/bw) ≈  0.00718479751373071

    @test floor(Int, bw) == 2

    for j in -floor(Int, bw):-1
        k_j = CovarianceMatrices.kernel(k, j/bw)
        LinearAlgebra.axpy!(k_j, CovarianceMatrices.Γ!(cache, j), cache.V)
    end

    for j in 1:floor(Int, bw)
        k_j = CovarianceMatrices.kernel(k, j/bw)
        LinearAlgebra.axpy!(k_j, CovarianceMatrices.Γ!(cache, j), cache.V)
    end

    LinearAlgebra.copytri!(cache.V, 'U')

    V = reshape([ 2.18378130064673,  -0.126247404742945, -0.0168728162431914,
                  -0.126247404742945, 17.4806466521012,    2.18514395521502,
                  -0.0168728162431914, 2.18514395521502,   0.273151430809999], 3, 3)

    @test all(V .≈ cache.V)


    fill!(cache.Q, zero(eltype(cache.Q)))
    for i = 1:size(cache.Q, 2)
        cache.Q[i,i] = one(eltype(cache.Q))
    end
    v = ldiv!(qr(I-cache.D'), cache.Q)
    cache.V .= v*cache.V*v'

    V = reshape([10.3056593241458,   7.62922768968777,  0.731955791380014,
                 7.62922768968773,  8.49755610886395,  0.859802385984295,
                 0.731955791380006, 0.859802385984297, 0.0874594553955033], 3, 3)

    @test all(V .≈ cache.V)
end

andrews_kernels = [:TruncatedKernel, :ParzenKernel, :TukeyHanningKernel, :QuadraticSpectralKernel, :BartlettKernel]
neweywest_kernels = [:ParzenKernel, :QuadraticSpectralKernel, :BartlettKernel]

@testset "HAC Univariate Long Run Variance................." begin
    univariate = JSON.parse(read("testdata/univariate.json", String))

    global x = [0.164199142438412, -0.22231320001538, -2.29596418288347,
         -0.562717239408665, 1.11832433510705, 0.259810317798567,
         0.647885100553029, 2.53438209392891, -0.419561292138475,
         1.19138801574376, 2.52661839907567, -0.443382492040113,
         -0.137169008509379, -0.967782699857501, 0.150507152028812,
         -1.27098181862663, 1.38639734998711, 0.231229342441316,
         -0.943827510026301, -1.11211457440442]

    function fopt!(u)
        global da = Dict{String, Any}()
        global dn = Dict{String, Any}()
        for pre in (:false, :true)
            da["bwtype"] = "auto"
            da["prewhite"] = pre == :true ? 1 : 0
            dn["bwtype"] = "auto"
            dn["prewhite"] = pre == :true ? 1 : 0
            for k in andrews_kernels
                eval(quote
                     tmp = CovarianceMatrices.covariance(x[1:end,:], ($k)(prewhiten=$pre))
                     da[String($k)] = Dict{String, Any}("bw" => tmp.K.bw, "V" => tmp.V)
                     if Symbol($k) in neweywest_kernels
                     tmp = CovarianceMatrices.covariance(x[1:end,:], ($k)(prewhiten=$pre))
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
            da["prewhite"] = pre == :true ? 1 : 0
            dn["bwtype"] = "auto"
            dn["prewhite"] = pre == :true ? 1 : 0
            for k in andrews_kernels
                eval(quote
                     tmp = CovarianceMatrices.covariance(x[1:end,:], ($k)(1.5, prewhiten=$pre))
                     da[String($k)] = Dict{String, Any}("bw" => tmp.K.bw, "V" => tmp.V)
                     if Symbol($k) in neweywest_kernels
                         tmp = CovarianceMatrices.covariance(x[1:end,:], ($k)(1.5, prewhiten=$pre))
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
        @test univariate[j][h][k]["V"] ≈ u[j][h][k]["V"]
        @test univariate[j][h][k]["bw"] ≈ u[j][h][k]["bw"]
    end

end

@testset "HAC Multivariate Long Run Variance..............." begin
    multivariate = JSON.parse(read("testdata/multivariate.json", String))

    global X = reshape([-0.392343481403554, 0.369702177262769, -0.283239954622452, -1.71955159119934,
                 -0.196311837779395, 0.567935573633728, 0.675556050609011, -0.59266740997335,
                 0.433501752110958, -0.108134518170885, 0.686698254275096,
                 0.905166380758529, 0.997306068740448, -0.498167979332402, -0.52789448547848,
                 0.100420689892425, 1.49495246674266, -0.601579368927597, -0.166121087348934,
                 0.545348413754815, 0.294595939676804, -1.4327640501377, 0.719755536433108,
                 -1.2570984424727, -1.5357895599007, 1.0027040264714, 1.08593183743541,
                 -0.711342086638968, -0.772154611498373, 1.30668004120162, 2.89166460485415,
                 0.614941151434828, -1.59157215625112, -0.51709522643053, 1.97543651877893,
                 1.92940239570577, 0.889679654045449, -0.471409046820902, -1.30435924644088,
                 0.424385975482526, -1.63421078193559, -0.562676024339334, -1.9773880104471,
                 -1.13903619779294, 0.586555209142932, -1.60335681344097, -1.19189517293108,
                 2.13456541115109, -1.42078068631219, -0.207019328929601, -0.600736667890819,
                 -1.41872438508684, -0.608094864262906, 1.07318477908557, -0.477244503174433,
                 0.567645883530209, -0.149728929145769, 1.41918460373266, 0.462399751718563,
                 -0.132320093478005, 1.27839375300926, -0.480093346919616, -0.0428876036353262,
                 -1.56004471316687, -0.134394994431227, 2.04942053641657, -1.8022396128532,
                 -1.72537103051051, -0.657108341488561, 1.34392729540088, 1.90019159830845,
                 0.126395933995686, -0.770308826277006, 0.457784471260252, -0.160271362465108,
                 0.0598471594810112, 1.75048422563306, -0.737657566749125, -0.462941254536989,
                 0.699940101308202, 1.2689352813542, -0.336296482224711, 1.42781944149594,
                 -1.69159952993731, -1.15816200645816, 0.83309270822555, -1.34670872662577,
                 2.24540522326547, 1.14409536415596, -0.959417691691381], 30, 3)

    function fopt!(u)
        global da = Dict{String, Any}()
        global dn = Dict{String, Any}()
        for pre in (:false, :true)
            da["bwtype"] = "auto"
            da["prewhite"] = pre == :true ? 1 : 0
            dn["bwtype"] = "auto"
            dn["prewhite"] = pre == :true ? 1 : 0
            for k in andrews_kernels
                eval(quote
                     tmp = CovarianceMatrices.covariance(X, ($k)(prewhiten=$pre))
                     da[String($k)] = Dict{String, Any}("bw" => tmp.K.bw, "V" => tmp.V)
                     if Symbol($k) in neweywest_kernels
                     tmp = CovarianceMatrices.covariance(X, ($k)(prewhiten=$pre))
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
            da["prewhite"] = pre == :true ? 1 : 0
            dn["bwtype"] = "auto"
            dn["prewhite"] = pre == :true ? 1 : 0
            for k in andrews_kernels
                eval(quote
                     tmp = CovarianceMatrices.covariance(X, ($k)(1.5, prewhiten=$pre))
                     da[String($k)] = Dict{String, Any}("bw" => tmp.K.bw, "V" => tmp.V)
                     if Symbol($k) in neweywest_kernels
                         tmp = CovarianceMatrices.covariance(X, ($k)(1.5, prewhiten=$pre))
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
        @test hcat(multivariate[j][h][k]["V"]...) ≈ u[j][h][k]["V"]
        @test multivariate[j][h][k]["bw"] ≈ u[j][h][k]["bw"]
    end

end

@testset "HAC OLS VCOV....................................." begin
    reg = JSON.parse(read("testdata/regression.json", String))
    global df = CSV.read("testdata/ols_df.csv")

    function fopt!(u)
        global da = Dict{String, Any}()
        global dn = Dict{String, Any}()
        for pre in (:false, :true)
            da["bwtype"] = "auto"
            da["prewhite"] = pre == :true ? 1 : 0
            dn["bwtype"] = "auto"
            dn["prewhite"] = pre == :true ? 1 : 0
            for k in andrews_kernels
                eval(quote
                     ols = glm(@formula(y~x1+x2+x3), df, Normal(), IdentityLink())
                     tmp = CovarianceMatrices.vcov(ols, ($k)(prewhiten=$pre), dof_adjustment = false)
                     da[String($k)] = Dict{String, Any}("bw" => tmp.K.bw, "V" => tmp.V)
                     if Symbol($k) in neweywest_kernels
                     tmp = CovarianceMatrices.vcov(ols, ($k)(prewhiten=$pre), dof_adjustment = false)
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
            da["prewhite"] = pre == :true ? 1 : 0
            dn["bwtype"] = "auto"
            dn["prewhite"] = pre == :true ? 1 : 0
            for k in andrews_kernels
                eval(quote
                     ols = glm(@formula(y~x1+x2+x3), df, Normal(), IdentityLink())
                     tmp = CovarianceMatrices.vcov(ols, ($k)(1.5, prewhiten=$pre), dof_adjustment = false)
                     da[String($k)] = Dict{String, Any}("bw" => tmp.K.bw, "V" => tmp.V)
                     if Symbol($k) in neweywest_kernels
                         tmp = CovarianceMatrices.vcov(ols, ($k)(1.5, prewhiten=$pre), dof_adjustment = false)
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



@testset "Accessor functions .............................." begin
    Random.seed!(1)
    y = randn(100)
    X = [ones(100) randn(100,4)]
    df = DataFrame(y = y, x1 = X[:,2], x2 = X[:,3], x3 = X[:,4], x4 = X[:,5])

    OLS1 = fit(GeneralizedLinearModel, X, y, Normal(), IdentityLink())
    OLS2 = fit(LinearModel, X, y)
    OLS3 = glm(@formula(y~x1+x2+x3+x4), df, Normal(), IdentityLink())
    OLS4 = lm(X,y)

    @test CovarianceMatrices.modelmatrix(OLS1) == X
    @test CovarianceMatrices.modelmatrix(OLS2) == X
    @test CovarianceMatrices.modelmatrix(OLS3) == X
    @test CovarianceMatrices.modelmatrix(OLS4) == X

    @test CovarianceMatrices.unweighted_nobs(OLS1) == 100
    @test CovarianceMatrices.unweighted_nobs(OLS1) == CovarianceMatrices.unweighted_nobs(OLS2)
    @test CovarianceMatrices.unweighted_nobs(OLS2) == CovarianceMatrices.unweighted_nobs(OLS3)
    @test CovarianceMatrices.unweighted_nobs(OLS3) == CovarianceMatrices.unweighted_nobs(OLS4)

    @test CovarianceMatrices.pseudohessian(OLS1) ≈ inv(X'X)
    @test CovarianceMatrices.pseudohessian(OLS2) ≈ inv(X'X)
    @test CovarianceMatrices.pseudohessian(OLS3) ≈ inv(X'X)
    @test CovarianceMatrices.pseudohessian(OLS4) ≈ inv(X'X)

    res = y - X*coef(OLS1)
    @test CovarianceMatrices.residuals(OLS1) ≈ res
    @test CovarianceMatrices.residuals(OLS2) ≈ res
    @test CovarianceMatrices.residuals(OLS3) ≈ res
    @test CovarianceMatrices.residuals(OLS4) ≈ res

end

@testset "HAC - GLM........................................" begin
    clotting = DataFrame(
        u    = log.([5,10,15,20,30,40,60,80,100]),
        lot1 = [118,58,42,35,27,25,21,19,18],
        lot2 = [69,35,26,21,18,16,13,12,12],
        w    = 9.0*[1/8, 1/9, 1/25, 1/6, 1/14, 1/25, 1/15, 1/13, 0.3022039]
    )

    GAMMA = glm(@formula(lot1~u), clotting, Gamma(),InverseLink(), wts = convert(Array, clotting[:w]))
    V = vcov(GAMMA, ParzenKernel())
    Vp = [5.48898e-7 -2.60409e-7; -2.60409e-7 1.4226e-7]
    @test V ≈ Vp atol = 1e-08

    GAMMA = glm(@formula(lot1~u), clotting, Gamma(),InverseLink())
    V = vcov(GAMMA, ParzenKernel())
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
    y = clotting[:lot1]
    GL  = fit(GeneralizedLinearModel, X,y, Normal(), IdentityLink())
    LM  = lm(X,y)

    S0 = vcov(OLS, HC0())
    S1 = vcov(OLS, HC1())
    S2 = vcov(OLS, HC2())
    S3 = vcov(OLS, HC3())
    S4 = vcov(OLS, HC4())
    S4m = vcov(OLS, HC4m())
    S5 = vcov(OLS, HC5())

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

    S0 = vcov(GL, HC0())
    S1 = vcov(GL, HC1())
    S2 = vcov(GL, HC2())
    S3 = vcov(GL, HC3())
    S4 = vcov(GL, HC4())
    S4m = vcov(GL, HC4m())
    S5 = vcov(GL, HC5())

    @test S0 ≈ St0
    @test S1 ≈ St1
    @test S2 ≈ St2
    @test S3 ≈ St3
    @test S4 ≈ St4
    @test S4m ≈ St4m
    @test S5 ≈ St5

    ## Weighted OLS though GLM interface
    wOLS = fit(GeneralizedLinearModel, @formula(lot1~u), clotting, Normal(),
               IdentityLink(), wts = Vector{Float64}(clotting[:w]))

    wts = Vector{Float64}(clotting[:w])
    X = [fill(1,size(clotting[:u])) clotting[:u]]
    y = clotting[:lot1]
    wLM = lm(X, y)
    wGL = fit(GeneralizedLinearModel, X, y, Normal(),
              IdentityLink(), wts = wts)


    S0 = vcov(wOLS, HC0())
    S1 = vcov(wOLS, HC1())
    S2 = vcov(wOLS, HC2())
    S3 = vcov(wOLS, HC3())
    S4 = vcov(wOLS, HC4())
    S4m= vcov(wOLS, HC4m())
    S5 = vcov(wOLS, HC5())



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

    S0 = vcov(GAMMA, HC0())
    S1 = vcov(GAMMA, HC1())
    S2 = vcov(GAMMA, HC2())
    S3 = vcov(GAMMA, HC3())
    S4 = vcov(GAMMA, HC4())
    S4m = vcov(GAMMA, HC4m())
    S5 = vcov(GAMMA, HC5())

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

    @test S0  ≈ St0 atol = 1e-06
    @test S1  ≈ St1 atol = 1e-06
    @test S2  ≈ St2 atol = 1e-06
    @test S3  ≈ St3 atol = 1e-06
    @test S4  ≈ St4 atol = 1e-05
    @test S4m ≈ St4m atol = 1e-06
    @test S5  ≈ St5 atol = 1e-06

    ## Weighted Gamma

    GAMMA = glm(@formula(lot1~u), clotting, Gamma(),InverseLink(), wts = convert(Array, clotting[:w]))

    S0 = vcov(GAMMA, HC0())
    S1 = vcov(GAMMA, HC1())
    S2 = vcov(GAMMA, HC2())
    S3 = vcov(GAMMA, HC3())
    S4 = vcov(GAMMA, HC4())
    S4m = vcov(GAMMA, HC4m())
    S5 = vcov(GAMMA, HC5())

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

    @test S0  ≈ St0 atol = 1e-06
    @test S1  ≈ St1 atol = 1e-06
    @test S2  ≈ St2 atol = 1e-06
    @test S3  ≈ St3 atol = 1e-06
    @test S4  ≈ St4 atol = 1e-05
    @test S4m ≈ St4m atol = 1e-06
    @test S5  ≈ St5 atol = 1e-06
end


@testset "CRHC............................................." begin

    df = CSV.read("testdata/wols_test.csv")
    df_sorted = sort!(copy(df), :cl)

    St1 = [.0374668, .0497666, .0472636, .0437952, .0513613, .0435369]

    OLS = fit(GeneralizedLinearModel, @formula(Y~X1+X2+X3+X4+X5), df, Normal(), IdentityLink())
    cl = convert(Array, df[:cl])
    k0 = CRHC0(cl)
    k1 = CRHC1(cl)
    k2 = CRHC2(cl)
    k3 = CRHC3(cl)


    V0 = vcov(OLS, k0)
    V1 = vcov(OLS, k1)
    V2 = vcov(OLS, k2)
    V3 = vcov(OLS, k3)

    @test V1 ≈ [0.00140376 0.000215526 -5.99768e-5 0.000296271 0.000460622 -0.000139741;
                0.000215526 0.00247671 -0.000270429 0.000218622 0.000610127 7.23345e-5;
                -5.99768e-5 -0.000270429 0.00223385 -0.000145166 -0.00018859 -0.000903561;
                0.000296271 0.000218622 -0.000145166 0.00191802 -0.000444364 -0.000420563;
                0.000460622 0.000610127 -0.00018859 -0.000444364 0.00263798 0.000736363;
                -0.000139741 7.23345e-5 -0.000903561 -0.000420563 0.000736363 0.00189546] atol = 1e-08

    ## Note sandwich in R has HC3 without G/(G-1) and CRHC2 is problematic

    @test stderror(OLS, k0, sorted = false) == sqrt.(diag(V0))

    OLS_sorted = fit(GeneralizedLinearModel, @formula(Y~X1+X2+X3+X4+X5), df_sorted, Normal(), IdentityLink())
    V0s = vcov(OLS_sorted, k0, sorted = true)
    V1s = vcov(OLS_sorted, k1, sorted = true)
    V2s = vcov(OLS_sorted, k2, sorted = true)
    V3s = vcov(OLS_sorted, k3, sorted = true)

    @test V0s == V0
    @test V1s == V1
    @test V2s == V2
    @test V3s == V3


    wOLS = fit(GeneralizedLinearModel, @formula(Y~X1+X2+X3+X4+X5), df,
               Normal(), IdentityLink(), wts = convert(Array{Float64}, df[:w]))

    cl = convert(Array, df[:cl])
    k0 = CRHC0(cl)
    k1 = CRHC1(cl)
    k2 = CRHC2(cl)
    k3 = CRHC3(cl)

    V0 = vcov(wOLS, k0)
    V1 = vcov(wOLS, k1)
    V2 = vcov(wOLS, k2)
    V3 = vcov(wOLS, k3)

    @test V1 ≈ [0.00183525 0.000137208 -0.00038971 0.000389943 0.000619903 0.00019496;
                0.000137208 0.00242781 -0.000272316 0.000462353 2.99597e-5 0.000133303;
                -0.00038971 -0.000272316 0.00273479 -0.000113765 -7.26396e-5 -0.000998524;
                0.000389943 0.000462353 -0.000113765 0.00171538 -0.00067357 -0.000416268;
                0.000619903 2.99597e-5 -7.26396e-5 -0.00067357 0.00225446 0.00106796;
                0.00019496 0.000133303 -0.000998524 -0.000416268 0.00106796 0.00226444] atol = 1e-07

    wOLS_sorted = fit(GeneralizedLinearModel, @formula(Y~X1+X2+X3+X4+X5), df_sorted,
                      Normal(), IdentityLink(), wts = convert(Array{Float64}, df[:w]))

    V0s = vcov(wOLS_sorted, k0, sorted = true)
    V1s = vcov(wOLS_sorted, k1, sorted = true)
    V2s = vcov(wOLS_sorted, k2, sorted = true)
    V3s = vcov(wOLS_sorted, k3, sorted = true)

    @test V1 ≈ [0.00183525 0.000137208 -0.00038971 0.000389943 0.000619903 0.00019496;
                0.000137208 0.00242781 -0.000272316 0.000462353 2.99597e-5 0.000133303;
                -0.00038971 -0.000272316 0.00273479 -0.000113765 -7.26396e-5 -0.000998524;
                0.000389943 0.000462353 -0.000113765 0.00171538 -0.00067357 -0.000416268;
                0.000619903 2.99597e-5 -7.26396e-5 -0.00067357 0.00225446 0.00106796;
                0.00019496 0.000133303 -0.000998524 -0.000416268 0.00106796 0.00226444] atol = 1e-07

    innovation = CSV.read("testdata/InstInnovation.csv", allowmissing=:none)

    innovation[:capemp] = log.(innovation[:capital]./innovation[:employment])
    innovation[:lsales] = log.(innovation[:sales])
    innovation[:year] = categorical(innovation[:year])
    innovation[:industry] = categorical(innovation[:industry])
    #innovation[:company] = categorical(innovation[:company])
    pois = glm(@formula(cites ~ institutions + capemp + lsales + industry + year), innovation, Poisson(), LogLink())

    Vt = [0.817387, 5.7907e-6, 0.0184833, 0.00172419]
    @test diag(vcov(pois, CRHC0(innovation[:company]), Matrix))[1:4] ≈ Vt atol = 1e-5

end


@testset "CovarianceMatrices Methods......................." begin
    df = CSV.read("testdata/wols_test.csv")
    df_sorted = sort!(copy(df), :cl)
    cl = convert(Array, df[:cl])
    wOLS = fit(GeneralizedLinearModel, @formula(Y~X1+X2+X3+X4+X5), df,
               Normal(), IdentityLink(), wts = convert(Array{Float64}, df[:w]))

    V0 = vcov(wOLS, HC0())
    V1 = vcov(wOLS, ParzenKernel())
    V2 = vcov(wOLS, CRHC0(cl))

    V0s = vcov(wOLS, HC0(), CovarianceMatrix, SVD)
    V1s = vcov(wOLS, ParzenKernel(), CovarianceMatrix, SVD)
    V2s = vcov(wOLS, CRHC0(cl), CovarianceMatrix, SVD)

    V0m = vcov(wOLS, HC0(), Matrix)
    V1m = vcov(wOLS, ParzenKernel(), Matrix)
    V2m = vcov(wOLS, CRHC0(cl), Matrix)
    
    @test V0 == V0m
    @test V1 == V1m
    @test V2 == V2m

    @test V0.F == cholesky(Symmetric(V0m))
    @test V1.F == cholesky(Symmetric(V1m))
    @test V2.F == cholesky(Symmetric(V2m))

    @test V0s.F == svd(V0m)
    @test V1s.F == svd(V1m)
    @test V2s.F == svd(V2m)

end