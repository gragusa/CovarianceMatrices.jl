using CovarianceMatrices
using Random
using GLM
using DataFrames
using BenchmarkTools

Random.seed!(1234)


u = zeros(6000*50)
for j = 1:2999
    u[j+1] = 0.97*u[j] + randn()
end

y = randn(300*50) .+ u[15001:30000]
df = DataFrame(randn(length(y), 5), :auto)
df[!, :y] .= y
df[!, :cluster] = repeat(1:150, inner = [100])

y = sqrt(2) .* randn(250)
df2 = DataFrame(randn(length(y), 5), :auto)
for j in Symbol.("x" .* string.(collect(1:5)))
    df2[!, j] = randn(250)
end
df2[!, :cluster] = repeat(1:50, inner = [5])
df2[!, :y] .= y

frm = @formula(y ~ x1 + x2 + x3 + x4 + x5)
lm1 = glm(frm, df, Normal(), IdentityLink())
lm2 = glm(frm, df2, Normal(), IdentityLink())

SUITE = BenchmarkGroup()
SUITE["HAC Andrews"] = BenchmarkGroup()
SUITE["HAC Newey"] = BenchmarkGroup()
SUITE["HAC Fixed(10)"] = BenchmarkGroup()
SUITE["HAC Fixed(30)"] = BenchmarkGroup()

SUITE["CRHC (large)"] = BenchmarkGroup()
SUITE["CRHC (small)"] = BenchmarkGroup()

SUITE["HAC Andrews"]["Parzen"] = @benchmarkable vcov($Parzen{Andrews}(), lm1)
SUITE["HAC Andrews"]["Truncated"] = @benchmarkable vcov($Truncated{Andrews}(), lm1)
SUITE["HAC Andrews"]["Bartlett"] = @benchmarkable vcov($Bartlett{Andrews}(), lm1)

SUITE["HAC Newey"]["Parzen"] = @benchmarkable vcov($Parzen{NeweyWest}(), lm1)
SUITE["HAC Newey"]["Bartlett"] = @benchmarkable vcov($Bartlett{NeweyWest}(), lm1)

SUITE["HAC Fixed(10)"]["Parzen"] = @benchmarkable vcov($Parzen(10), lm1)
SUITE["HAC Fixed(10)"]["Truncated"] = @benchmarkable vcov($Truncated(10), lm1)
SUITE["HAC Fixed(10)"]["Bartlett"] = @benchmarkable vcov($Bartlett(10), lm1)

SUITE["HAC Fixed(30)"]["Parzen"] = @benchmarkable vcov($Parzen(30), lm1)
SUITE["HAC Fixed(30)"]["Truncated"] = @benchmarkable vcov($Truncated(30), lm1)
SUITE["HAC Fixed(30)"]["Bartlett"] = @benchmarkable vcov($Bartlett(30), lm1)

k0 = CR0(df[!, :cluster])
k2 = CR2(df[!, :cluster])
k3 = CR3(df[!, :cluster])

k20 = CR0(df2[!, :cluster])
k22 = CR2(df2[!, :cluster])
k23 = CR3(df2[!, :cluster])

SUITE["CRHC (large)"]["CRHC0"] = @benchmarkable vcov($k0, lm1)
SUITE["CRHC (large)"]["CRHC2"] = @benchmarkable vcov($k2, lm1)
SUITE["CRHC (large)"]["CRHC3"] = @benchmarkable vcov($k3, lm1)

SUITE["CRHC (small)"]["CRHC0"] = @benchmarkable vcov($k20, lm2)
SUITE["CRHC (small)"]["CRHC2"] = @benchmarkable vcov($k22, lm2)
SUITE["CRHC (small)"]["CRHC3"] = @benchmarkable vcov($k23, lm2)
