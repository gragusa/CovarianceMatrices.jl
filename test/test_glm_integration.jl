@testset "GLM Integration (HAC weights)" begin
    rng = StableRNG(1234)
    df = DataFrame(X1 = randn(rng, 10), X2 = randn(rng, 10), Y = randn(rng, 10))
    m = lm(@formula(Y ~ X1 + X2), df)
    M = momentmatrix(m)
    @test M ≈ [-0.44514751 -0.23125171  0.391807752;
                             -0.04562477 -0.04129679 -0.076279862;
                             -0.71538836  1.19874849  0.104035222;
                              0.08256957 -0.10690558 -0.033658338;
                              0.59596497 -0.40785784  0.318478340;
                              0.49708819 -0.34107733 -0.401033205;
                              1.05956405  0.09773035  0.446067578;
                             -1.14182761  0.10235007 -0.769137251;
                             -0.12131377  0.17616015  0.007348568;
                              0.23411525 -0.44659979  0.012371195]
    K = Bartlett{NeweyWest}()
    Σ = a𝕍ar(K, m)
    @test K.bw[1] ≈ 1.737626 atol = 1e-06
    ## Notice that using Σ = a𝕍ar(K, M) give a different bandwidth
    Σ_M = a𝕍ar(K, M)
    @test K.bw[1] ≈ 0.3638178431924936 atol = 1e-06
    ## The reason is the weights are all one
    @test K.kw == [1.0, 1.0, 1.0]
    ## This can be overridden by setting the weights manually
    K.kw .= [0.0, 1.0, 1.0]
    ## and locking the kernel so that the weights are not changed
    K.wlock .= true
    Σ_M = a𝕍ar(K, M)
    @test K.bw[1] ≈ 1.737626 atol = 1e-06
    @test K.wlock[1] === true
end

@testset "GLM Integration (_residuals)" begin
    ## GLM `residuals` function is sketchy at best, so we provide our own
    ## _residuals
    ##  - linear models: (y - Xβ).*sqrt(wts)
    ## - GLM: working residuals .* sqrt(wts)
    using StableRNGs
    using DataFrames
    using RCall
    rng = StableRNG(1234)
    df = DataFrame(X1 = randn(rng, 20), X2 = randn(rng, 20), Y = randn(rng, 20), w = rand(rng, 20), cl = repeat(1:4, inner = 5))
    m = lm(@formula(Y ~ X1 + X2), df)
    r = CovarianceMatrices._residuals(m.model)
    @rput df
    R"""
    library(sandwich)
    m <- lm(Y ~ X1 + X2, data = df)
    resid <- residuals(m)
    """
    @rget resid
    @test r ≈ resid
    R"""
    library(sandwich)
    m <- lm(Y ~ X1 + X2, data = df, weights = w)
    resid <- residuals(m)
    """
    @rget resid
    m = lm(@formula(Y ~ X1 + X2), df, wts = df.w)
    r = CovarianceMatrices._residuals(m.model)
    @test r ≈ resid.*sqrt.(df.w)
    R"""
    Mr <- sandwich::estfun(m) # working residuals
    """
    @rget Mr
    M = momentmatrix(m)
    @test M ≈ Mr

    R"""
    library(sandwich)
    V0 <- vcovHC(m, type = "HC0")
    V1 <- vcovHC(m, type = "HC1")
    V2 <- vcovHC(m, type = "HC2")
    V3 <- vcovHC(m, type = "HC3")
    V4 <- vcovHC(m, type = "HC4")
    V5 <- vcovHC(m, type = "HC5")
    """
    @rget V0 V1 V2 V3 V4 V5

    Σ0 = vcov(HC0(), m)
    Σ1 = vcov(HC1(), m)
    Σ2 = vcov(HC2(), m)
    Σ3 = vcov(HC3(), m)
    Σ4 = vcov(HC4(), m)
    Σ5 = vcov(HC5(), m)

    @test Σ0 ≈ V0 rtol = 1e-6
    @test Σ1 ≈ V1 rtol = 1e-6
    @test Σ2 ≈ V2 rtol = 1e-6
    @test Σ3 ≈ V3 rtol = 1e-6
    @test Σ4 ≈ V4 rtol = 1e-6
    @test Σ5 ≈ V5 rtol = 1e-6

    ## GLM
    R"""
     counts <- c(18,17,15,20,10,20,25,13,12)
     outcome <- gl(3,1,9)
     treatment <- gl(3,3)
     df = data.frame(treatment, outcome, counts) # showing data
     glm.D93 <- glm(counts ~ outcome + treatment, family = poisson())
     Mr <- sandwich::estfun(glm.D93)
    V0 <- vcovHC(glm.D93, type = "HC0")
    V1 <- vcovHC(glm.D93, type = "HC1")
    V2 <- vcovHC(glm.D93, type = "HC2")
    V3 <- vcovHC(glm.D93, type = "HC3")
    V4 <- vcovHC(glm.D93, type = "HC4")
    V5 <- vcovHC(glm.D93, type = "HC5")
     """
    @rget df Mr V0 V1 V2 V3 V4 V5
    glmD93 = glm(@formula(counts ~ outcome + treatment), df, Poisson())
    Σ0 = vcov(HC0(), glmD93)
    Σ1 = vcov(HC1(), glmD93)
    Σ2 = vcov(HC2(), glmD93)
    Σ3 = vcov(HC3(), glmD93)
    Σ4 = vcov(HC4(), glmD93)
    Σ5 = vcov(HC5(), glmD93)
    @test momentmatrix(glmD93) ≈ Mr atol = 1e-6
    @test Σ0 ≈ V0 rtol = 1e-6
    @test Σ1 ≈ V1 rtol = 1e-6
    @test Σ2 ≈ V2 rtol = 1e-6
    @test Σ3 ≈ V3 rtol = 1e-6
    @test Σ4 ≈ V4 rtol = 1e-6
    @test Σ5 ≈ V5 rtol = 1e-6

    R"""
     counts <- c(18,17,15,20,10,20,25,13,12)
     outcome <- gl(3,1,9)
     treatment <- gl(3,3)
     df = data.frame(treatment, outcome, counts) # showing data
     df$weights <- c(1.0, 0.5, 1.5, 1.0, 0.8, 1.2, 1.0, 0.9, 1.1)
     glm.D93 <- glm(counts ~ outcome + treatment, family = poisson(), weights = df$weights)
     Mr <- sandwich::estfun(glm.D93)
    V0 <- vcovHC(glm.D93, type = "HC0")
    V1 <- vcovHC(glm.D93, type = "HC1")
    V2 <- vcovHC(glm.D93, type = "HC2")
    V3 <- vcovHC(glm.D93, type = "HC3")
    V4 <- vcovHC(glm.D93, type = "HC4")
    V5 <- vcovHC(glm.D93, type = "HC5")
     """
    @rget df Mr V0 V1 V2 V3 V4 V5
    glmD93 = glm(@formula(counts ~ outcome + treatment), df, Poisson(), wts=df.weights)
    Σ0 = vcov(HC0(), glmD93)
    Σ1 = vcov(HC1(), glmD93)
    Σ2 = vcov(HC2(), glmD93)
    Σ3 = vcov(HC3(), glmD93)
    Σ4 = vcov(HC4(), glmD93)
    Σ5 = vcov(HC5(), glmD93)
    @test momentmatrix(glmD93) ≈ Mr atol = 1e-6
    @test Σ0 ≈ V0 rtol = 1e-6
    @test Σ1 ≈ V1 rtol = 1e-6
    @test Σ2 ≈ V2 rtol = 1e-6
    @test Σ3 ≈ V3 rtol = 1e-6
    @test Σ4 ≈ V4 rtol = 1e-6
    @test Σ5 ≈ V5 rtol = 1e-6

    ## HAC
    R"""
    V0 <- kernHAC(glm.D93, kernel = "Bartlett", prewhite=FALSE, bw= bwAndrews, adjust=FALSE)
    V1 <- kernHAC(glm.D93, kernel = "Bartlett", prewhite=FALSE, bw= bwAndrews, adjust=TRUE)
    M0 <- meatHAC(glm.D93, kernel = "Bartlett", prewhite=FALSE, bw= bwAndrews)
     """
    @rget V0 V1 M0
    Σ₀ =  vcov(Bartlett{Andrews}(), glmD93, dofadjust = false)
    Σ₁ =  vcov(Bartlett{Andrews}(), glmD93, dofadjust = true)
    @test Σ₀ ≈ V0 rtol = 1e-6
    @test Σ₁ ≈ V1 rtol = 1e-6
    ## GLM
    rng = StableRNGs.StableRNG(1234)
    df = DataFrame(count = abs.(round.(Int, 10.0*randn(rng, 100))), X2 = randn(rng, 100))
    dum = glm(@formula(count ~ X2), df, Poisson())
    @rput df
    R"""
     dum <- glm(count ~ X2, data = df, family = poisson())
     V0 <- kernHAC(dum, kernel = "Bartlett", prewhite=FALSE, bw= bwAndrews, adjust=FALSE)
    V1 <- kernHAC(dum, kernel = "Bartlett", prewhite=FALSE, bw= bwAndrews, adjust=TRUE)
    M0 <- meatHAC(dum, kernel = "Bartlett", prewhite=FALSE, bw= bwAndrews)
    """
    @rget V0 V1 M0
    vcov0 = vcov(Bartlett{Andrews}(), dum, dofadjust = false)
    vcov1 = vcov(Bartlett{Andrews}(), dum, dofadjust = true)
    k = Bartlett{Andrews}()
    vcov0 = vcov(k, dum, dofadjust = false)
    k.wlock .= true
    aVar(k, dum)

        clotting = DataFrame(
            u = log.([5, 10, 15, 20, 30, 40, 60, 80, 100]),
            lot1 = [118, 58, 42, 35, 27, 25, 21, 19, 18],
            lot2 = [69, 35, 26, 21, 18, 16, 13, 12, 12],
            w = 9.0*[1/8, 1/9, 1/25, 1/6, 1/14, 1/25, 1/15, 1/13, 0.3022039]
        )

        GAMMA = glm(
            @formula(lot1~u),
            clotting,
            Gamma(),
            InverseLink(),
            wts = convert(Array, clotting[!, :w])
        )
        k = Parzen{Andrews}()
        V = vcov(k, GAMMA)
        bw = k.bw[1]
        M = aVar(k, GAMMA)
        @rput clotting
        R"""
        m <- glm(lot1~u, data=clotting, family=Gamma, weights=clotting$w)
        Vr = kernHAC(m, kernel="Parzen", bw = bwAndrews, prewhite=FALSE)
        bwr = sandwich::bwAndrews(m, prewhite=F, kernel = "Parzen")
        Mr = meatHAC(m, kernel = "Parzen", prewhite = F, bw = bwAndrews)
        MM <- sandwich::estfun(m)
        """
        @rget Vr bwr Mr MM
        @test MM ≈ momentmatrix(GAMMA)  atol = 1e-08
        @test bwr ≈ bw atol = 1e-04
        @test M ≈ Mr atol = 1e-08


        Vp = [5.48898e-7 -2.60409e-7; -2.60409e-7 1.4226e-7]
        @test V ≈ Vp atol = 1e-08

        GAMMA = glm(@formula(lot1~u), clotting, Gamma(), InverseLink())
        k = Parzen{Andrews}()
        V = vcov(k, GAMMA)
        Vp = [5.81672e-7 -2.24162e-7; -2.24162e-7 1.09657e-7]
        @test V ≈ Vp atol = 1e-08
    end

end
