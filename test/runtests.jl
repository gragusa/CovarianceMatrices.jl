using VCOV
using Base.Test
using DataFrames
using GLM

# write your own tests here
X = randn(2040, 5);

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
    lot2 = [69,35,26,21,18,16,13,12,12]
)

OLS = glm(lot1~u,clotting, Normal(), IdentityLink())
Ω = meat(OLS.model, HC0())
Q = bread(OLS.model)
Σ = vcov(OLS, HC0())

Qt = [13.382804946157550674 -3.741352244575458119;
	  -3.741352244575458119  1.130415659364268688]

Ωt = [206.6102758242464006  518.7871394793446598; 
	  518.7871394793446598  1531.1729257437255001]

Σt = [720.6213064108161461 -190.06451254313066102;
	  -190.0645125431306042   51.16333374239605547]

@test abs(maximum(Ω .- Ωt)) < 1e-12
@test abs(maximum(Q .- Qt)) < 1e-12
@test abs(maximum(Σ .- Σt)) < 1e-12

GAMMA = glm(lot1~u, clotting, Gamma(),InverseLink())

Σt = [4.504287921232951e-07 -1.700020601541489e-07;
	  -1.700020601541490e-07  8.203697048568913e-08]

Σ = vcov(GAMMA, HC0())	  
@test abs(maximum(Σ .- Σt)) < 1e-12

