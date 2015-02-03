using CovarianceMatrices
using Base.Test

############################################################
## HAC
############################################################

X = randn(100, 5);

@time vcov(X, TruncatedKernel(2.))
@time vcov(X, BartlettKernel(2.))
@time vcov(X, ParzenKernel(2.))
@time vcov(X, QuadraticSpectralKernel(2.))
@time vcov(X, TruncatedKernel())
@time vcov(X, BartlettKernel())
@time vcov(X, ParzenKernel())
@time vcov(X, QuadraticSpectralKernel())

############################################################
## HC
############################################################

# A Gamma example, from McCullagh & Nelder (1989, pp. 300-2)
clotting = DataFrame(
    u    = log([5,10,15,20,30,40,60,80,100]),
    lot1 = [118,58,42,35,27,25,21,19,18],
    lot2 = [69,35,26,21,18,16,13,12,12],
    w    = [1/8, 1/9, 1/25, 1/6, 1/14, 1/25, 1/15, 1/13, 0.3022039]
)

## Unweighted OLS though GLM interface
OLS = fit(GeneralizedLinearModel, lot1~u,clotting, Normal(), IdentityLink())
S0 = vcov(OLS, HC0())
S1 = vcov(OLS, HC1())
S2 = vcov(OLS, HC2())
S3 = vcov(OLS, HC3())
S4 = vcov(OLS, HC4())
S4m = vcov(OLS, HC4m())
S5 = vcov(OLS, HC5())

St0 = [720.6213 -190.0645; -190.0645 51.16333]
St1 = [926.5131 -244.3687; -244.3687 65.78143]
St2 = [1300.896 -343.3307; -343.3307 91.99719]
St3 = [2384.504 -628.975 ; -628.975 167.7898]
St4 = [2538.746 -667.9597; -667.9597 177.2631]
St4m= [3221.095 -849.648 ; -849.648 226.1705]
St5 = St4

@test abs(maximum(S0 .- St0)) < 1e-04
@test abs(maximum(S1 .- St1)) < 1e-04
@test abs(maximum(S2 .- St2)) < 1e-04
@test abs(maximum(S3 .- St3)) < 1e-04
@test abs(maximum(S4 .- St4)) < 1e-03
@test abs(maximum(S4m .- St4m)) < 1e-03
@test abs(maximum(S5 .- St5)) < 1e-03

M0 = meat(OLS.model, HC0())
M1 = meat(OLS.model, HC1())
M2 = meat(OLS.model, HC2())
M3 = meat(OLS.model, HC3())
M4 = meat(OLS.model, HC4())
M4m = meat(OLS.model, HC4m())
M5 = meat(OLS.model, HC5())

Mt0 = [206.6103 518.7871; 518.7871 1531.173]
Mt1 = [265.6418 667.012;  667.012 1968.651]
Mt2 = [323.993 763.0424;  763.0424 2149.767]
Mt3 = [531.478 1172.751;  1172.751 3122.79]
Mt4 = [531.1047 1110.762; 1110.762 2783.269]
Mt4m= [669.8647 1412.227; 1412.227 3603.247]
Mt5 = Mt4

@test abs(maximum(M0 .- Mt0)) < 1e-04
@test abs(maximum(M1 .- Mt1)) < 1e-04
@test abs(maximum(M2 .- Mt2)) < 1e-04
@test abs(maximum(M3 .- Mt3)) < 1e-03
@test abs(maximum(M4 .- Mt4)) < 1e-03
@test abs(maximum(M4m .- Mt4m)) < 1e-03
@test abs(maximum(M5 .- Mt5)) < 1e-03

## Unweighted
OLS = glm(lot1~u,clotting, Normal(), IdentityLink())
S0 = vcov(OLS, HC0())
S1 = vcov(OLS, HC1())
S2 = vcov(OLS, HC2())
S3 = vcov(OLS, HC3())
S4 = vcov(OLS, HC4())

## Weighted OLS though GLM interface
wOLS = fit(GeneralizedLinearModel, lot1~u,clotting, Normal(),
           IdentityLink(), wts = array(clotting[:w]))
S0 = vcov(wOLS, HC0())
S1 = vcov(wOLS, HC1())
S2 = vcov(wOLS, HC2())
S3 = vcov(wOLS, HC3())
S4 = vcov(wOLS, HC4())
S4m= vcov(wOLS, HC4m())
S5 = vcov(wOLS, HC5())

St0 = [717.7362 -178.4043; -178.4043 45.82273]
St1 = [922.8037 -229.3769; -229.3769 58.91494]
St2 = [1412.94 -361.33; -361.33 95.91252]
St3 = [2869.531 -756.2976; -756.2976 208.2344]
St4 = [3969.913 -1131.358; -1131.358 342.2859]
St4m= [4111.626 -1103.174; -1103.174   310.194]
St5 = St4

@test abs(maximum(S0 .- St0)) < 1e-04
@test abs(maximum(S1 .- St1)) < 1e-04
@test abs(maximum(S2 .- St2)) < 1e-03
@test abs(maximum(S3 .- St3)) < 1e-03
@test abs(maximum(S4 .- St4)) < 1e-03
@test abs(maximum(S4m .- St4m)) < 1e-03
@test abs(maximum(S5 .- St5)) < 1e-03

## Unweighted GLM - Gamma
GAMMA = glm(lot1~u, clotting, Gamma(),InverseLink())

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

St4m= [8.49306e-05 -2.43618e-05; -2.43618e-05  7.04210e-06]

St5 = St4
       
@test abs(maximum(S0 .- St0)) < 1e-06
@test abs(maximum(S1 .- St1)) < 1e-06
@test abs(maximum(S2 .- St2)) < 1e-06
@test abs(maximum(S3 .- St3)) < 1e-06
@test abs(maximum(S4 .- St4)) < 1e-06
@test abs(maximum(S4m .- St4m)) < 1e-05
@test abs(maximum(S5 .- St5)) < 1e-05

## Weighted Gamma

GAMMA = glm(lot1~u, clotting, Gamma(),InverseLink(), wts = array(clotting[:w]))

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

St4m = [8.493064e-05 -2.436180e-05; -2.436180e-05  7.042101e-06]

St5 = St4


@test abs(maximum(S0 .- St0)) < 1e-06
@test abs(maximum(S1 .- St1)) < 1e-06
@test abs(maximum(S2 .- St2)) < 1e-06
@test abs(maximum(S3 .- St3)) < 1e-06
@test abs(maximum(S4 .- St4)) < 1e-06
@test abs(maximum(S4m .- St4m)) < 1e-05
@test abs(maximum(S5 .- St5)) < 1e-05


### Cluster basic interface

## Construct Fake ols data

srand(1)

df = DataFrame( Y = randn(500),
               X1 = randn(500),
               X2 = randn(500),
               X3 = randn(500),
               X4 = randn(500),
               X5 = randn(500),
               w  = rand(500),
               cl = repmat([1:25], 20))

OLS = fit(GeneralizedLinearModel, Y~X1+X2+X3+X4+X5, df,
          Normal(), IdentityLink())


S0 = vcov(OLS, HC0())
S1 = vcov(OLS, HC1())
S2 = vcov(OLS, HC2())
S3 = vcov(OLS, HC3())
S4 = vcov(OLS, HC4())
S5 = vcov(OLS, HC5())

cl = array(df[:cl])
S0 = stderr(OLS, CRHC0(cl))
S1 = stderr(OLS, CRHC1(cl))
S2 = stderr(OLS, CRHC2(cl))
S3 = stderr(OLS, CRHC3(cl))


## STATA
St1 = [.0374668, .0497666, .0472636, .0437952, .0513613, .0435369]

@test maximum(abs(S0 .- St1)) < 1e-02
@test maximum(abs(S1 .- St1)) < 1e-04
@test maximum(abs(S2 .- St1)) < 1e-02
@test maximum(abs(S3 .- St1)) < 1e-02


wOLS = fit(GeneralizedLinearModel, Y~X1+X2+X3+X4+X5, df,
          Normal(), IdentityLink(), wts = array(df[:w]))

S0 = stderr(wOLS, CRHC0(cl))
S1 = stderr(wOLS, CRHC1(cl))
S2 = stderr(wOLS, CRHC2(cl))
S3 = stderr(wOLS, CRHC3(cl))

St1 = [0.042839848169137905,0.04927285387211425,
       0.05229519531359171,0.041417170723876025,
       0.04748115282615204,0.04758615959662984]

@test maximum(abs(S1 .- St1)) < 1e-10



############################################################
## Instrumental Variables
############################################################

using ModelsGenerators

srand(1)
y, x, z = randiv(n = 500, k = 3, m = 15);
cl = repmat([1:25], 20)
ww = rand(500)

iivv = ivreg(x,z,reshape(y, 500))

mut = [0.18402423026700765,-0.18069781054811473,-0.10666434926774973,
       0.1716025154131859,-0.0794834970843437,0.1728389042278303,
       -0.19455903354864845,-0.04328538690391654,0.3573882542662863]

mu = predict(iivv)
@test maximum(mu[1:9] - mut)<1e-10

ut = [0.06743220393049221,-0.33965003523101844,-0.2636323254480044,
      0.41789439202294165,0.09369677251530516,1.357234945075299,
      -1.1479591516308787,-0.34120715739919666,0.012435837904900104]

u = residuals(iivv)
@test maximum(u[1:9] - ut) < 1e-10


betat = [0.26050875643847554]
@test maximum(betat - coef(iivv)) < 1e-09

@test_approx_eq stderr(iivv) [0.15188237754059922]
@test_approx_eq stderr(iivv, HC1()) [0.16119631706495913]
@test_approx_eq stderr(iivv, HC2()) [0.16159542810823724]
@test_approx_eq stderr(iivv, HC3()) [0.16216172762549416]
@test_approx_eq stderr(iivv, HC4()) [0.16295579476374497]
@test_approx_eq stderr(iivv, HC4m()) [0.16242050216392523]
@test_approx_eq stderr(iivv, HC5()) [0.16437294390550466]

iivv = ivreg(x,z,reshape(y, 500), wts = ww)

@test_approx_eq sqrt(vcov(iivv, HC1()))  [0.16634761896913675]
@test_approx_eq sqrt(vcov(iivv, HC3()))  [0.16786988598366934]


@test_approx_eq sqrt(vcov(iivv, CRHC0(cl)))  [0.17650498286330474]
@test_approx_eq sqrt(vcov(iivv, CRHC1(cl)))  [0.18014464378074402]
@test_approx_eq sqrt(vcov(iivv, CRHC2(cl)))  [0.18100382220522054]
@test_approx_eq sqrt(vcov(iivv, CRHC3(cl)))  [0.18961072793672662]




srand(1)
y, x, z = randiv(n = 2500, k = 3, m = 15);
add_x = randn(2500, 20)
x = [x add_x]
z = [z add_x]
cl = repmat([1:50], 50)
ww = rand(2500)

println("Timing of ivreg")
@time iivv = ivreg(x,z,reshape(y, 2500), wts = ww)
