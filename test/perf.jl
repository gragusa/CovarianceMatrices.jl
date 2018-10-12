using CovarianceMatrices
using Statistics
using CSV
using BenchmarkTools


X = randn(10000,20);
k = BartlettKernel(prewhiten=true)
cfg = CovarianceMatrices.HACConfig(X, k)
CovarianceMatrices.variance(X, k, cfg)

@benchmark CovarianceMatrices.variance(X, k, cfg, calculatechol=false)
