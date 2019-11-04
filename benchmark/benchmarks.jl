using CovarianceMatrices
using Random
using GLM
using DataFrames
using BenchmarkTools

Random.seed!(1)


u = zeros(6000*50)
for j in 1:2999
    u[j+1] = 0.97*u[j] + randn()
end

df = DataFrame(y = randn(300*50) .+ u[15001:30000])

df2 = DataFrame(y = sqrt(2).*randn(250))
for j in Symbol.("x".*string.(collect(1:5)))
    df2[j] = randn(250)
end
df2[:cluster] = repeat(1:50, inner = [5])


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

SUITE["HAC Andrews"]["Parzen"] = @benchmarkable vcov($ParzenKernel(), lm1)
SUITE["HAC Andrews"]["Truncated"] = @benchmarkable vcov($TruncatedKernel(), lm1)
SUITE["HAC Andrews"]["Bartlett"] = @benchmarkable vcov($BartlettKernel(), lm1)

SUITE["HAC Newey"]["Parzen"] = @benchmarkable vcov($ParzenKernel(NeweyWest), lm1)
SUITE["HAC Newey"]["Bartlett"] = @benchmarkable vcov($BartlettKernel(NeweyWest), lm1)

SUITE["HAC Fixed(10)"]["Parzen"] = @benchmarkable vcov($ParzenKernel(10), lm1)
SUITE["HAC Fixed(10)"]["Truncated"] = @benchmarkable vcov($TruncatedKernel(10), lm1)
SUITE["HAC Fixed(10)"]["Bartlett"] = @benchmarkable vcov($BartlettKernel(10), lm1)

SUITE["HAC Fixed(30)"]["Parzen"] = @benchmarkable vcov($ParzenKernel(30), lm1)
SUITE["HAC Fixed(30)"]["Truncated"] = @benchmarkable vcov($TruncatedKernel(30), lm1)
SUITE["HAC Fixed(30)"]["Bartlett"] = @benchmarkable vcov($BartlettKernel(30), lm1)

k0 = CRHC0(df[!,:cluster])
k2 = CRHC2(df[!,:cluster])
k3 = CRHC3(df[!,:cluster])

k20 = CRHC0(df2[!,:cluster])
k22 = CRHC2(df2[!,:cluster])
k23 = CRHC3(df2[!,:cluster])

SUITE["CRHC (large)"]["CRHC0"] = @benchmarkable vcov($k0, lm1)
SUITE["CRHC (large)"]["CRHC2"] = @benchmarkable vcov($k2, lm1)
SUITE["CRHC (large)"]["CRHC3"] = @benchmarkable vcov($k3, lm1)

SUITE["CRHC (small)"]["CRHC0"] = @benchmarkable vcov($k20, lm2)
SUITE["CRHC (small)"]["CRHC2"] = @benchmarkable vcov($k22, lm2)
SUITE["CRHC (small)"]["CRHC3"] = @benchmarkable vcov($k23, lm2)