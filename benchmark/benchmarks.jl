using CovarianceMatrices
using Random
using GLM
using DataFrames
using BenchmarkTools

Random.seed!(1)

df = DataFrame(y = randn(1000))
for j in Symbol.("x".*string.(collect(1:5)))
    df[j] = randn(1000)
end

frm = @formula(y ~ x1 + x2 + x3 + x4 + x5)
lm1 = glm(frm, df, Normal(), IdentityLink())
k_fix = TruncatedKernel(2)
k_opt = TruncatedKernel()

suite = BenchmarkGroup()
suite["HAC"] = BenchmarkGroup(["Optimal Uncached", "Fixed Uncached"])
suite["HAC"]["Optimal Uncached"] = @benchmarkable vcov(lm1, $k_opt)
suite["HAC"]["Fixed Uncached"] = @benchmarkable vcov(lm1, $k_fix)
