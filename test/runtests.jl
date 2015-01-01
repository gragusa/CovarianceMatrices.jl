using CovarianceMatrices
using Base.Test
using DataFrames
using GLM

# write your own tests here
X = randn(100, 5);

@time vcov(X, TruncatedKernel(2.))
@time vcov(X, BartlettKernel(2.))
@time vcov(X, ParzenKernel(2.))
@time vcov(X, QuadraticSpectralKernel(2.))

@time vcov(X, TruncatedKernel())
@time vcov(X, BartlettKernel())
@time vcov(X, ParzenKernel())
@time vcov(X, QuadraticSpectralKernel())


# A Gamma example, from McCullagh & Nelder (1989, pp. 300-2)
clotting = DataFrame(
    u    = log([5,10,15,20,30,40,60,80,100]),
    lot1 = [118,58,42,35,27,25,21,19,18],
    lot2 = [69,35,26,21,18,16,13,12,12],
    w    = [1/8, 1/9, 1/25, 1/6, 1/14, 1/25, 1/15, 1/13, 0.3022039]
)

wOLS = fit(GeneralizedLinearModel, lot1~u,clotting, Normal(), IdentityLink(), wts = array(clotting[:w]))
Σ = vcov(wOLS, HC0())
Σt = [717.7361780762186 -178.40427498072842;
      -178.4042749807284   45.82273069735402]
@test abs(maximum(Σ .- Σt)) < 1e-08

Σ1 = vcov(wOLS, HC1())
Σ2 = vcov(wOLS, HC2())
Σ3 = vcov(wOLS, HC3())
Σ4 = vcov(wOLS, HC4())


Σ1t = [922.8036575265651 -229.37692497522175;
	   -229.3769249752218   58.91493946802645]

Σ2t = [1412.9404975839295 -361.32996934499948;
		-361.3299693449993   95.91252069568421]

Σ3t = [2869.530686899533 -756.2976102694070;
	   -756.297610269407  208.2343786897033]

Σ4t = [3969.913026297301 -1131.3577578048037;
	   -1131.357757804804   342.2858663028415]


@test abs(maximum(Σ1 .- Σ1t)) < 1e-08
@test abs(maximum(Σ2 .- Σ2t)) < 1e-08
@test abs(maximum(Σ3 .- Σ3t)) < 1e-08
@test abs(maximum(Σ4 .- Σ4t)) < 1e-08


# methods(ModelMatrix)
# names(wOLS)
# ModelMatrix(wOLS.model.pp)
# XX = ModelMatrix(wOLS.mf)
# inv(cholfact(wOLS.model.pp))
# ww = wOLS.model.rr.wts
# methods(wts)
# inv((XX.*sqrt(ww))'*(XX.*sqrt(ww)))

OLS = glm(lot1~u,clotting, Normal(), IdentityLink())
Ω = meat(OLS.model, HC0())
Q = bread(OLS.model)
Σ = vcov(OLS, HC0())
Σ1 = vcov(OLS, HC1())
Σ2 = vcov(OLS, HC2())
Σ3 = vcov(OLS, HC3())
Σ4 = vcov(OLS, HC4())

Qt = [13.382804946157550674 -3.741352244575458119;
	  -3.741352244575458119  1.130415659364268688]

Ωt = [206.6102758242464006  518.7871394793446598;
	  518.7871394793446598  1531.1729257437255001]

Σt = [720.6213064108161461 -190.06451254313066102;
	  -190.0645125431306042   51.16333374239605547]

@test abs(maximum(Ω .- Ωt)) < 1e-08
@test abs(maximum(Q .- Qt)) < 1e-08
@test abs(maximum(Σ .- Σt)) < 1e-08

Σ1t = [926.5131082424773 -244.36865898402502;
	   -244.3686589840250   65.78142909736637]

Σ2t = [1300.8956732838612 -343.33067269940670;
	   -343.3306726994066   91.99718603308419]

Σ3t = [2384.5039334663807 -628.9749923203956;
	   -628.9749923203955  167.7897687767359]

Σ4t = [2538.7463538446455 -667.9597231907179;
	   -667.9597231907181  177.2630895747554]

@test abs(maximum(Σ1 .- Σ1t)) < 1e-08
@test abs(maximum(Σ2 .- Σ2t)) < 1e-08
@test abs(maximum(Σ3 .- Σ3t)) < 1e-08
@test abs(maximum(Σ4 .- Σ4t)) < 1e-08


GAMMA = glm(lot1~u, clotting, Gamma(),InverseLink())

Σ = vcov(GAMMA, HC0())
Σt = [4.504287921232951e-07 -1.700020601541489e-07;
	  -1.700020601541490e-07  8.203697048568913e-08]
@test abs(maximum(Σ .- Σt)) < 1e-08

Σ1 = vcov(GAMMA, HC1())
Σ1t = [5.791227327299548e-07 -2.185740773410504e-07;
	   -2.185740773410510e-07  1.054761049101728e-07]
@test abs(maximum(Σ1 .- Σ1t)) < 1e-08

Σ2 = vcov(GAMMA, HC2())
Σ2t = [3.192633083111232e-06 -9.942484630848573e-07;
	   -9.942484630848578e-07  3.329973305723091e-07]
@test abs(maximum(Σ2 .- Σ2t)) < 1e-08

Σ3 = vcov(GAMMA, HC3())
Σ3t = [2.982697811926944e-05 -8.948137019946751e-06;
       -8.948137019946738e-06  2.712024459305714e-06]
@test abs(maximum(Σ3 .- Σ3t)) < 1e-08

Σ4 = vcov(GAMMA, HC4())
Σ4t = [0.002840158946368653 -0.0008474436578800430;
      -0.000847443657880045  0.0002528819761961959]
@test abs(maximum(Σ4 .- Σ4t)) < 1e-08
