using PkgBenchmark
using CovarianceMatrices
using Random
using GLM
using DataFrames
using PkgBenchmark
Random.seed!(1)

df = DataFrame(y = randn(1000))

for j in Symbol.("x".*string.(collect(1:5)))
    df[j] = randn(1000)
end

frm = @formula(y ~ x1 + x2 + x3 + x4 + x5)
lm1 = glm(frm, df, Normal(), IdentityLink())

bench = BenchmarkGroup()

suite["HAC"] = BenchmarkGroup(["Optimal Cached"], ["Optimal Uncached"], ["Fixed Cached"], ["Fixed Uncached"])

suite["HAC"]["Optimal Uncached"] = @benchmarkable vcov(lm1, TruncatedKernel())
suite["HAC"]["Fixed Uncached"] = @benchmarkable vcov(lm1, TruncatedKernel(2))
# ["Fixed Cached"]
# ["Fixed Uncached"]


# @benchgroup "HAC - Fixed Bandwidth" begin
#     @benchgroup "No prewithen" begin
#         nf = string(Pkg.dir("CovarianceMatrices"), "/test/ols_hac.csv")
#         df = CSV.read(nf)
# 	      lm1 = glm(@formula(y~x+w), df, Normal(), IdentityLink())
#         @bench "Truncated Kernel" vcov(lm1, TruncatedKernel(1.0), prewhite = false)
#         @bench "Quadratic Spectral Kernel" vcov(lm1, QuadraticSpectralKernel(1.0), prewhite = false)
#         @bench "Parzen Kernel" vcov(lm1, ParzenKernel(1.0), prewhite = false)
#         @bench "Tukey Hanning Kernel" vcov(lm1, TukeyHanningKernel(1.0), prewhite = false)
#     end
#
#     @benchgroup "Prewithen" begin
#         nf = string(Pkg.dir("CovarianceMatrices"), "/test/ols_hac.csv")
# 	      df = CSV.read(nf)
#         lm1 = glm(@formula(y~x+w), df, Normal(), IdentityLink())
#         @bench "Truncated Kernel" vcov(lm1, TruncatedKernel(1.0), prewhite = true)
#         @bench "Quadratic Spectral Kernel" vcov(lm1, QuadraticSpectralKernel(1.0), prewhite = true)
#         @bench "Parzen Kernel" vcov(lm1, ParzenKernel(1.0), prewhite = true)
#         @bench "Tukey Hanning Kernel" vcov(lm1, TukeyHanningKernel(1.0), prewhite = true)
#     end
#
# end
#
#
# @benchgroup "HAC - Optimal Bandwidth" begin
#     @benchgroup "No prewithen" begin
#         nf = string(Pkg.dir("CovarianceMatrices"), "/test/ols_hac.csv")
#         df = CSV.read(nf)
#         lm1 = glm(@formula(y~x+w), df, Normal(), IdentityLink())
#         @bench "Truncated Kernel" vcov(lm1, TruncatedKernel(), prewhite = false)
#         @bench "Quadratic Spectral Kernel" vcov(lm1, QuadraticSpectralKernel(), prewhite = false)
#         @bench "Parzen Kernel" vcov(lm1, ParzenKernel(), prewhite = false)
#         #@bench "Tukey Hanning Kernel" vcov(lm1, TukeyHanningKernel(), prewhite = false)
#     end
#
#     @benchgroup "Prewithen" begin
#         nf = string(Pkg.dir("CovarianceMatrices"), "/test/ols_hac.csv")
# 	      df = CSV.read(nf)
#         lm1 = glm(@formula(y~x+w), df, Normal(), IdentityLink())
#         @bench "Truncated Kernel" vcov(lm1, TruncatedKernel(), prewhite = true)
#         @bench "Quadratic Spectral Kernel" vcov(lm1, QuadraticSpectralKernel(), prewhite = true)
#         @bench "Parzen Kernel" vcov(lm1, ParzenKernel(), prewhite = true)
#         #@bench "Tukey Hanning Kernel" vcov(lm1, TukeyHanningKernel(), prewhite = true)
#     end
# end
