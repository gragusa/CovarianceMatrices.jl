using CovarianceMatrices
using Random
using GLM
using DataFrames

Random.seed!(1)

u = zeros(6000*50)
for j in 1:2999
    u[j+1] = 0.97*u[j] + randn()
end

df = DataFrame(y = randn(300*50) .+ u[15001:30000])

for j in Symbol.("x".*string.(collect(1:5)))
    df[!, j] = randn(300*50)
end
df[!, :cluster] = repeat(1:50, inner = [300])

df2 = DataFrame(y = sqrt(2).*randn(250))
for j in Symbol.("x".*string.(collect(1:5)))
    df2[!, j] .= randn(250)
end
df2[!, :cluster] .= repeat(1:50, inner = [5])

frm = @formula(y ~ x1 + x2 + x3 + x4 + x5)
lm1 = glm(frm, df, Normal(), IdentityLink())
lm2 = glm(frm, df2, Normal(),IdentityLink())

using Profile
Profile.clear()
@profile for j in 1:200; vcov(CRHC0(df[!,:cluster]), lm1); end
Juno.profiler()
using LinearAlgebra
k0 = CRHC0(df[!,:cluster])
cache = CovarianceMatrices.install_cache(k0, lm1)

using BenchmarkTools
@btime CovarianceMatrices.__vcov(k0, lm1, cache, Matrix, Cholesky, CovarianceMatrices.dofadjustment(k0, cache))

k2 = CRHC2(df[!,:cluster])
cache = CovarianceMatrices.install_cache(k2, lm1)
@btime CovarianceMatrices.__vcov(k2, lm1, cache, Matrix, Cholesky, CovarianceMatrices.dofadjustment(k2, cache))
