using CovarianceMatrices
using Test
using CSV
using LinearAlgebra
using Statistics

datapath = joinpath(@__DIR__)

@testset "HAC - Basic checks...................................." begin
    X = [0 0; 3 7; 4 8; 5 9]
    X_demean = [-3  -6; 0   1; 1   2; 2   3]
    cfg = CovarianceMatrices.HACConfig(X, TruncatedKernel(prewhiten=true));
    CovarianceMatrices.demean!(cfg, X, Val{true})
    @test all(cfg.μ .== [3 6])
    @test all(cfg.X_demean .== X_demean)
    CovarianceMatrices.fit_var!(cfg)
    @test all(cfg.D   .≈ [-9/5 -4; 1 2])
    @test all(cfg.XX  .≈ [3/5 1; 0 0; 9/5 3])
end

@testset "HAC - Optimal Bandwidth Calculations.................." begin
    X = [1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7,  .8,
         .3, .4, .5, .6, .7, .8, .9, .10, .11, .12]
    X = reshape(X, 8, 3)

    k = ParzenKernel(prewhiten=true)
    cfg = CovarianceMatrices.HACConfig(X, k)
    CovarianceMatrices.demean!(cfg, X, Val{true})

    Xd = reshape([               -2.75,    -1.75,   -0.75,     0.25,
                                  1.25,     2.25,    3.25,    -1.75,
                    -0.312500000000002,   0.6875,   1.6875,  2.6875,
                                3.6875,  -2.5125,  -3.0125, -2.9125,
                    0.0212499999999998,  0.12125,  0.22125, 0.32125,
                               0.42125, -0.37875, -0.36875, -0.35875], 8, 3)

    @test all(Xd .≈ cfg.X_demean)

    CovarianceMatrices.prewhiten!(cfg)

    D= reshape([-0.409214603090833,   -0.546926852567603, -0.0793279159360713,
                  7.17805137100394,    -1.09268085748226,  -0.152674132870786,
                  -56.8116543172651,    12.0360552297997,    1.52457939755027], 3,3)'

    @test all(D .≈ cfg.D)

    XX = reshape([0.575048549180823,     0.487377212994233,   0.399705876807639,
                  0.312034540621058,      0.22436320443446,   0.688172853937608,
                  0.254529685703059,     -1.41377778615736,  0.0222244009125362,
                  1.45822658798243,       2.89422877505233,   -2.86976903787778,
                  0.0313806821395586,   0.0116065536680327,   -0.17700974754426,
                  0.00253436150757022,     0.1820784705594,    0.36162257961123,
                  -0.35883331133694,   0.00357849884047592, 0.00132355436565176], 7, 3)
    @test all(XX .≈ cfg.XX)

    CovarianceMatrices.fit_ar!(cfg)
    ρ = [-0.411560904449045,  -0.202815142161935, -0.202567957968907]
    σ⁴= [ 0.00043730337156934, 8.49515993964198,   0.0020740008652368]
    @test all(ρ .≈ cfg.ρ)
    @test all(σ⁴ .≈ cfg.σ⁴)

    CovarianceMatrices.fit_ar!(cfg)
    if isempty(k.weights)
        for j in 1:size(X,2)
            push!(k.weights, 1.0)
        end
    end
    a1, a2 = CovarianceMatrices.getalpha!(cfg, k.weights)
    @test a1 ≈ 0.1789771071933
    @test a2 ≈ 0.07861018601427
    @test CovarianceMatrices.bw_andrews(k, a1, a2, 7) ≈ 2.361704327253
    k.bw .= CovarianceMatrices.bw_andrews(k, a1, a2, 7)
    bw = k.bw[1]
    fill!(cfg.V, zero(eltype(cfg.V)))
    mul!(cfg.V, cfg.XX', cfg.XX)
    V = reshape([0.707026904730492,   0.0322404268557179, 0.00367653990459955,
                 0.0322404268557179, 10.3694701634646,    1.29616765021965,
                 0.00367653990459955, 1.29616765021965,   0.162019118007504], 3, 3)*2

    @test all(V ≈ cfg.V)
    triu!(cfg.V)
    @test CovarianceMatrices.kernel(k, 1/bw) ≈  0.379763254768776
    @test CovarianceMatrices.kernel(k, 2/bw) ≈  0.00718479751373071

    @test floor(Int, bw) == 2

    for j in -floor(Int, bw):-1
        k_j = CovarianceMatrices.kernel(k, j/bw)
        LinearAlgebra.axpy!(k_j, CovarianceMatrices.Γ!(cfg, j), cfg.V)
    end

    for j in 1:floor(Int, bw)
        k_j = CovarianceMatrices.kernel(k, j/bw)
        LinearAlgebra.axpy!(k_j, CovarianceMatrices.Γ!(cfg, j), cfg.V)
    end

    LinearAlgebra.copytri!(cfg.V, 'U')

    V = reshape([ 2.18378130064673,  -0.126247404742945, -0.0168728162431914,
                 -0.126247404742945, 17.4806466521012,    2.18514395521502,
                 -0.0168728162431914, 2.18514395521502,   0.273151430809999], 3, 3)

    @test all(V .≈ cfg.V)


    fill!(cfg.Q, zero(eltype(cfg.Q)))
    for i = 1:size(cfg.Q, 2)
        cfg.Q[i,i] = one(eltype(cfg.Q))
    end
    v = ldiv!(qr(I-cfg.D'), cfg.Q)
    cfg.V .= v*cfg.V*v'

    V = reshape([10.3056593241458,   7.62922768968777,  0.731955791380014,
          7.62922768968773,  8.49755610886395,  0.859802385984295,
          0.731955791380006, 0.859802385984297, 0.0874594553955033], 3, 3)

    @test all(V .≈ cfg.V)




end

@testset "HAC - Optimal Bandwidth (All Kernels/All bw).........." begin

    X = [1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7,  .8,
        .3, .4, .5, .6, .7, .8, .9, .10, .11, .12]
    X = reshape(X, 8, 3)

    h_pre = CovarianceMatrices.HACConfig(X, ParzenKernel(prewhiten=true))
    h_unw = CovarianceMatrices.HACConfig(X, ParzenKernel(prewhiten=false))

    andrews_opt_kernels = (((ParzenKernel(prewhiten=u), u ? h_pre : h_unw),
                            (TruncatedKernel(prewhiten=u), u ? h_pre : h_unw),
                            (BartlettKernel(prewhiten=u), u ? h_pre : h_unw),
                            (TukeyHanningKernel(prewhiten=u), u ? h_pre : h_unw),
                            (QuadraticSpectralKernel(prewhiten=u), u ? h_pre : h_unw)) for u in (true, false))


    Ω = map(lst -> map(k -> CovarianceMatrices.variance(X, k...), lst), andrews_opt_kernels)

    O = (([1.28821 0.953653 0.0914945; 0.953653 1.06219 0.107475; 0.0914945 0.107475 0.0109324],
         [1.49062 1.1894 0.115455; 1.1894 1.18561 0.118517; 0.115455 0.118517 0.0118871],
         [1.3913 1.07462 0.103798; 1.07462 1.12689 0.113294; 0.103798 0.113294 0.0114393],
         [1.34422 1.0202 0.0982714; 1.0202 1.09906 0.110818; 0.0982714 0.110818 0.011227],
         [1.36003 1.05029 0.101445; 1.05029 1.12827 0.113729; 0.101445 0.113729 0.011515]),

         ([4.70394 -2.91171 -0.450844; -2.91171 6.48574 0.850683; -0.450844 0.850683 0.112987],
         [6.95313 -2.61172 -0.461328; -2.61172 10.8807 1.38987; -0.461328 1.38987 0.180094],
         [5.80137 -2.247 -0.38614; -2.247 9.16004 1.15495; -0.38614 1.15495 0.147795],
         [5.44375 -3.05791 -0.48333; -3.05791 7.51892 0.979098; -0.48333 0.979098 0.129227],
         [5.51144 -3.28821 -0.512786; -3.28821 7.52871 0.985677; -0.512786 0.985677 0.130714]))

    for j in 1:2, i in 1:4
        @test abs2(maximum(Ω[j][i] .- O[j][i])) < 1e-06
    end

    ## Quadratic Spectral in R has a bug
    newey_opt_kernels = (((ParzenKernel(NeweyWest, prewhiten=u), u ? h_pre : h_unw),
                            (BartlettKernel(NeweyWest, prewhiten=u), u ? h_pre : h_unw),
                            (QuadraticSpectralKernel(NeweyWest, prewhiten=u), u ? h_pre : h_unw)) for u in (true, false))

    Ω = map(lst -> map(k -> CovarianceMatrices.variance(X, k...), lst), newey_opt_kernels)

    O = (([0.623214 -0.105139 -0.0189412; -0.105139 0.244904 0.0285405; -0.0189412 0.0285405 0.00340343],
          [1.31044 0.981166 0.0943068; 0.981166 1.07909 0.109042; 0.0943068 0.109042 0.0110746],
          [0.629345 -0.0685547 -0.0149303; -0.0685547 0.219566 0.0252903; -0.0149303 0.0252903 0.00299439]),

         ([5.19829 -1.76977 -0.309974; -1.76977 8.62069 1.07022; -0.309974 1.07022 0.134827],
          [4.25189 -1.07127 -0.19531; -1.07127 6.67455 0.828456; -0.19531 0.828456 0.104354],
          [2.68879 -2.3731 -0.345941; -2.3731 3.53072 0.481831; -0.345941 0.481831 0.0662011]))

    for j in 1:2, i in 1:3
      @test abs2(maximum(Ω[j][i] .- O[j][i])) < 1e-07
    end

    W = randn(100,5);
    k = ParzenKernel(prewhiten=true)
    h_pre = CovarianceMatrices.HACConfig(W, k)
    Ω = CovarianceMatrices.variance(W, k,  h_pre, calculatechol = true)
    @test h_pre.chol == cholesky(Symmetric(Ω), check = false)
    ## Without cfg
    Ω = map(lst -> map(k -> CovarianceMatrices.variance(X, k[1]), lst), newey_opt_kernels)
    for j in 1:2, i in 1:3
      @test abs2(maximum(Ω[j][i] .- O[j][i])) < 1e-06
    end

end

#
#
#
# ############################################################
# ## HAC
# ############################################################
#
# X = randn(100, 5);
#
# @time vcov(X, TruncatedKernel(2.))
# @time vcov(X, BartlettKernel(2.))
# @time vcov(X, ParzenKernel(2.))
# @time vcov(X, QuadraticSpectralKernel(2.))
# @time vcov(X, TruncatedKernel())
# @time vcov(X, BartlettKernel())
# @time vcov(X, ParzenKernel())
# @time vcov(X, QuadraticSpectralKernel())
#
# ############################################################
# ## HC
# ############################################################
#
# # A Gamma example, from McCullagh & Nelder (1989, pp. 300-2)
# clotting = DataFrame(
#     u    = log.([5,10,15,20,30,40,60,80,100]),
#     lot1 = [118,58,42,35,27,25,21,19,18],
#     lot2 = [69,35,26,21,18,16,13,12,12],
#     w    = 9.0*[1/8, 1/9, 1/25, 1/6, 1/14, 1/25, 1/15, 1/13, 0.3022039]
# )
#
# ## Unweighted OLS though GLM interface
# OLS = fit(GeneralizedLinearModel, @formula(lot1~u),clotting, Normal(), IdentityLink())
# mf = ModelFrame(@formula(lot1~u),clotting)
# X = ModelMatrix(mf).m
# y = clotting[:lot1]
# GL  = fit(GeneralizedLinearModel, X,y, Normal(), IdentityLink())
# LM  = lm(X,y)
#
# S0 = vcov(OLS, HC0())
# S1 = vcov(OLS, HC1())
# S2 = vcov(OLS, HC2())
# S3 = vcov(OLS, HC3())
# S4 = vcov(OLS, HC4())
# S4m = vcov(OLS, HC4m())
# S5 = vcov(OLS, HC5())
#
# St0 = [720.6213 -190.0645; -190.0645 51.16333]
# St1 = [926.5131 -244.3687; -244.3687 65.78143]
# St2 = [1300.896 -343.3307; -343.3307 91.99719]
# St3 = [2384.504 -628.975 ; -628.975 167.7898]
# St4 = [2538.746 -667.9597; -667.9597 177.2631]
# St4m= [3221.095 -849.648 ; -849.648 226.1705]
# St5 = St4
#
# @test abs.(maximum(S0 .- St0)) < 1e-04
# @test abs.(maximum(S1 .- St1)) < 1e-04
# @test abs.(maximum(S2 .- St2)) < 1e-04
# @test abs.(maximum(S3 .- St3)) < 1e-04
# @test abs.(maximum(S4 .- St4)) < 1e-03
# @test abs.(maximum(S4m .- St4m)) < 1e-03
# @test abs.(maximum(S5 .- St5)) < 1e-03
#
# S0 = vcov(GL, HC0())
# S1 = vcov(GL, HC1())
# S2 = vcov(GL, HC2())
# S3 = vcov(GL, HC3())
# S4 = vcov(GL, HC4())
# S4m = vcov(GL, HC4m())
# S5 = vcov(GL, HC5())
#
# @test St0 ≈ S0 atol = 1e-4
# @test St1 ≈ S1 atol = 1e-4
# @test St2 ≈ S2 atol = 1e-3
# @test St3 ≈ S3 atol = 1e-4
# @test St4 ≈ S4 atol = 1e-3
# @test St4m ≈ S4m atol = 1e-3
# @test St5 ≈ S5 atol = 1e-3
#
# M0 = CovarianceMatrices.meat(OLS, HC0())
# M1 = CovarianceMatrices.meat(OLS, HC1())
# M2 = CovarianceMatrices.meat(OLS, HC2())
# M3 = CovarianceMatrices.meat(OLS, HC3())
# M4 = CovarianceMatrices.meat(OLS, HC4())
# M4m = CovarianceMatrices.meat(OLS, HC4m())
# M5 = CovarianceMatrices.meat(OLS, HC5())
#
# Mt0 = [206.6103 518.7871; 518.7871 1531.173]
# Mt1 = [265.6418 667.012;  667.012 1968.651]
# Mt2 = [323.993 763.0424;  763.0424 2149.767]
# Mt3 = [531.478 1172.751;  1172.751 3122.79]
# Mt4 = [531.1047 1110.762; 1110.762 2783.269]
# Mt4m= [669.8647 1412.227; 1412.227 3603.247]
# Mt5 = Mt4
#
# @test abs.(maximum(M0 .- Mt0)) < 1e-04
# @test abs.(maximum(M1 .- Mt1)) < 1e-04
# @test abs.(maximum(M2 .- Mt2)) < 1e-04
# @test abs.(maximum(M3 .- Mt3)) < 1e-03
# @test abs.(maximum(M4 .- Mt4)) < 1e-03
# @test abs.(maximum(M4m .- Mt4m)) < 1e-03
# @test abs.(maximum(M5 .- Mt5)) < 1e-03
#
# ## Unweighted
# OLS = glm(@formula(lot1~u),clotting, Normal(), IdentityLink())
# S0 = vcov(OLS, HC0())
# S1 = vcov(OLS, HC1())
# S2 = vcov(OLS, HC2())
# S3 = vcov(OLS, HC3())
# S4 = vcov(OLS, HC4())
#
# ## Weighted OLS though GLM interface
# wOLS = fit(GeneralizedLinearModel, @formula(lot1~u), clotting, Normal(),
#            IdentityLink(), wts = Vector{Float64}(clotting[:w]))
#
# wts = Vector{Float64}(clotting[:w])
# X = [fill(1,size(clotting[:u])) clotting[:u]]
# y = clotting[:lot1]
# wLM = lm(X, y)
# wGL = fit(GeneralizedLinearModel, X, y, Normal(),
#             IdentityLink(), wts = wts)
#
# residuals_raw = y-X*coef(wGL)
# residuals_wts = sqrt.(wts).*(y-X*coef(wGL))
#
# @test CovarianceMatrices.modelresiduals(wOLS) == CovarianceMatrices.modelresiduals(wGL)
# @test CovarianceMatrices.modelresiduals(wOLS) == residuals_wts
# @test CovarianceMatrices.modelresiduals(wGL)  == residuals_wts
# @test CovarianceMatrices.modelweights(wGL)    == CovarianceMatrices.modelweights(wOLS)
# @test CovarianceMatrices.rawresiduals(wGL)    == CovarianceMatrices.rawresiduals(wOLS)
#
# wXX   = (wts.*X)'*X
# wXu   = X.*(residuals_raw.*wts)
# wXuuX = wXu'*wXu
#
# @test CovarianceMatrices.fullyweightedmodelmatrix(wOLS) == X.*wts
# @test CovarianceMatrices.fullyweightedmodelmatrix(wOLS).*CovarianceMatrices.rawresiduals(wOLS) ≈ wXu
# @test CovarianceMatrices.meat(wOLS, HC0()) ≈  wXuuX./(sum(wts))
#
# @test CovarianceMatrices.XX(wOLS) ≈ wXX
# @test CovarianceMatrices.invXX(wOLS) ≈ inv(wXX)
#
# S0 = vcov(wOLS, HC0())
# S1 = vcov(wOLS, HC1())
# S2 = vcov(wOLS, HC2())
# S3 = vcov(wOLS, HC3())
# S4 = vcov(wOLS, HC4())
# S4m= vcov(wOLS, HC4m())
# S5 = vcov(wOLS, HC5())
#
# St0 = [717.7362 -178.4043; -178.4043 45.82273]
# St1 = [922.8037 -229.3769; -229.3769 58.91494]
# St2 = [1412.94 -361.33; -361.33 95.91252]
# St3 = [2869.531 -756.2976; -756.2976 208.2344]
# St4 = [3969.913 -1131.358; -1131.358 342.2859]
# St4m= [4111.626 -1103.174; -1103.174   310.194]
# St5 = St4
#
# @test abs.(maximum(S0 .- St0)) < 1e-04
# @test abs.(maximum(S1 .- St1)) < 1e-04
# @test abs.(maximum(S2 .- St2)) < 1e-03
# @test abs.(maximum(S3 .- St3)) < 1e-03
# @test abs.(maximum(S4 .- St4)) < 1e-03
# @test abs.(maximum(S4m .- St4m)) < 1e-03
# @test abs.(maximum(S5 .- St5)) < 1e-03
#
# ## Unweighted GLM - Gamma
# GAMMA = glm(@formula(lot1~u), clotting, Gamma(),InverseLink())
#
# S0 = vcov(GAMMA, HC0())
# S1 = vcov(GAMMA, HC1())
# S2 = vcov(GAMMA, HC2())
# S3 = vcov(GAMMA, HC3())
# S4 = vcov(GAMMA, HC4())
# S4m = vcov(GAMMA, HC4m())
# S5 = vcov(GAMMA, HC5())
#
# St0 = [4.504287921232951e-07 -1.700020601541489e-07;
#        -1.700020601541490e-07  8.203697048568913e-08]
#
# St1 = [5.791227327299548e-07 -2.185740773410504e-07;
#        -2.185740773410510e-07  1.054761049101728e-07]
#
# St2 = [3.192633083111232e-06 -9.942484630848573e-07;
#        -9.942484630848578e-07  3.329973305723091e-07]
#
# St3 = [2.982697811926944e-05 -8.948137019946751e-06;
#        -8.948137019946738e-06  2.712024459305714e-06]
#
# St4 = [0.002840158946368653 -0.0008474436578800430;
#        -0.000847443657880045  0.0002528819761961959]
#
# St4m= [8.49306e-05 -2.43618e-05; -2.43618e-05  7.04210e-06]
#
# St5 = St4
#
# @test abs.(maximum(S0 .- St0)) < 1e-06
# @test abs.(maximum(S1 .- St1)) < 1e-06
# @test abs.(maximum(S2 .- St2)) < 1e-06
# @test abs.(maximum(S3 .- St3)) < 1e-06
# @test abs.(maximum(S4 .- St4)) < 1e-06
# @test abs.(maximum(S4m .- St4m)) < 1e-05
# @test abs.(maximum(S5 .- St5)) < 1e-05
#
# ## Weighted Gamma
#
# GAMMA = glm(@formula(lot1~u), clotting, Gamma(),InverseLink(), wts = convert(Array, clotting[:w]))
#
# S0 = vcov(GAMMA, HC0())
# S1 = vcov(GAMMA, HC1())
# S2 = vcov(GAMMA, HC2())
# S3 = vcov(GAMMA, HC3())
# S4 = vcov(GAMMA, HC4())
# S4m = vcov(GAMMA, HC4m())
# S5 = vcov(GAMMA, HC5())
#
# St0 = [4.015104e-07 -1.615094e-07;
#        -1.615094e-07  8.378363e-08]
#
# St1 = [5.162277e-07 -2.076549e-07;
#        -2.076549e-07  1.077218e-07]
#
# St2 = [2.720127e-06 -8.490977e-07;
#        -8.490977e-07  2.963563e-07]
#
# St3 = [2.638128e-05 -7.639883e-06;
#        -7.639883e-06  2.259590e-06]
#
# St4 = [0.0029025754 -0.0008275858;
#        -0.0008275858  0.0002360053]
#
# St4m = [8.493064e-05 -2.436180e-05; -2.436180e-05  7.042101e-06]
#
# St5 = St4
#
#
# @test abs.(maximum(S0 .- St0)) < 1e-06
# @test abs.(maximum(S1 .- St1)) < 1e-06
# @test abs.(maximum(S2 .- St2)) < 1e-06
# @test abs.(maximum(S3 .- St3)) < 1e-06
# @test abs.(maximum(S4 .- St4)) < 1e-06
# @test abs.(maximum(S4m .- St4m)) < 1e-05
# @test abs.(maximum(S5 .- St5)) < 1e-05
#
#
# ### Cluster basic interface
#
# ## Construct Fake ols data
#
# ## srand(1)
#
# ## df = DataFrame( Y = randn(500),
# ##                X1 = randn(500),
# ##                X2 = randn(500),
# ##                X3 = randn(500),
# ##                X4 = randn(500),
# ##                X5 = randn(500),
# ##                w  = rand(500),
# ##                cl = repmat(collect(1:25), 20))
#
# df = CSV.read("wols_test.csv")
#
# OLS = fit(GeneralizedLinearModel, @formula(Y~X1+X2+X3+X4+X5), df,
#           Normal(), IdentityLink())
#
# S0 = vcov(OLS, HC0())
# S1 = vcov(OLS, HC1())
# S2 = vcov(OLS, HC2())
# S3 = vcov(OLS, HC3())
# S4 = vcov(OLS, HC4())
# S5 = vcov(OLS, HC5())
#
# cl = convert(Array, df[:cl])
# S0 = stderror(OLS, CRHC0(cl))
# S1 = stderror(OLS, CRHC1(cl))
# S2 = stderror(OLS, CRHC2(cl))
# S3 = stderror(OLS, CRHC3(cl))
#
#
# ## STATA
# St1 = [.0374668, .0497666, .0472636, .0437952, .0513613, .0435369]
#
# @test maximum(abs.(S0 .- St1)) < 1e-02
# @test maximum(abs.(S1 .- St1)) < 1e-04
# @test maximum(abs.(S2 .- St1)) < 1e-02
# @test maximum(abs.(S3 .- St1)) < 1e-02
#
#
# wOLS = fit(GeneralizedLinearModel, @formula(Y~X1+X2+X3+X4+X5), df,
#           Normal(), IdentityLink(), wts = convert(Array{Float64}, df[:w]))
#
# S0 = stderror(wOLS, CRHC0(cl))
# S1 = stderror(wOLS, CRHC1(cl))
# S2 = stderror(wOLS, CRHC2(cl))
# S3 = stderror(wOLS, CRHC3(cl))
#
# St1 = [0.042839848169137905,0.04927285387211425,
#        0.05229519531359171,0.041417170723876025,
#        0.04748115282615204,0.04758615959662984]
#
# @test maximum(abs.(S1 .- St1)) < 1e-10
#
# ############################################################
# ## Test different interfaces
# ############################################################
#
# # y = randn(100);
# # x = randn(100, 5);
#
# # lm1 = lm(x, y)
# # @test stderror(lm1, HC0())≈[0.0941998, 0.0946132, 0.0961678, 0.0960445, 0.101651] atol=1e-06
# # @test diag(vcov(lm1, HC0()))≈[0.0941998, 0.0946132, 0.0961678, 0.0960445, 0.101651].^2 atol=1e-06
#
# ############################################################
# ## HAC
# ############################################################
#
# include("ols_hac.jl")
