# AirspeedVelocity.jl benchmark suite. The `Benchmark` workflow runs this script
# against both the pull request and the base branch, so it may use only API that
# exists on both. Run locally with `julia benchmark/run_asv.jl`.

using BenchmarkTools
using CovarianceMatrices
using DataFrames
using GLM
using Random

const SUITE = BenchmarkGroup()

# Each benchmark gets at most `SECONDS` of sampling.
const SECONDS = 1.0

rng = Xoshiro(20261016)

# AR(1) moment matrix, T × k.
function ar1_matrix(rng, T, k; ρ = 0.6)
    X = randn(rng, T, k)
    for t in 2:T
        X[t, :] .+= ρ .* X[t - 1, :]
    end
    return X
end

# ---------------------------------------------------------------------------
# HAC on a moment matrix
# ---------------------------------------------------------------------------
X = ar1_matrix(rng, 10_000, 6)

SUITE["HAC"] = BenchmarkGroup()
for (name, k) in (
    "Bartlett(10)" => Bartlett(10),
    "Parzen(10)" => Parzen(10),
    "Truncated(10)" => Truncated(10),
    "QuadraticSpectral(10)" => QuadraticSpectral(10),
    "Bartlett{NeweyWest}" => Bartlett{NeweyWest}(),
    "Parzen{NeweyWest}" => Parzen{NeweyWest}(),
    "Bartlett{Andrews}" => Bartlett{Andrews}(),
    "Parzen{Andrews}" => Parzen{Andrews}(),
    "QuadraticSpectral{Andrews}" => QuadraticSpectral{Andrews}()
)
    SUITE["HAC"][name] = @benchmarkable aVar($k, $X) seconds=SECONDS
end
SUITE["HAC"]["Bartlett{Andrews} prewhite"] = @benchmarkable aVar(
    $(Bartlett{Andrews}()), $X; prewhite = true) seconds=SECONDS

# ---------------------------------------------------------------------------
# Other long-run variance estimators on a moment matrix
# ---------------------------------------------------------------------------
SUITE["LRV"] = BenchmarkGroup()
for (name, k) in (
    "EWC(10)" => EWC(10),
    "VARHAC()" => VARHAC(),
    "VARHAC(FixedLags(4))" => VARHAC(FixedLags(4)),
    "VARHAC DifferentOwnLags([4, 4])" => VARHAC(AICSelector(), DifferentOwnLags([4, 4])),
    "UniformSmoother(5)" => UniformSmoother(5),
    "TriangularSmoother(5)" => TriangularSmoother(5)
)
    SUITE["LRV"][name] = @benchmarkable aVar($k, $X) seconds=SECONDS
end

# Driscoll-Kraay on a balanced panel: 50 units observed over 200 periods.
tis = repeat(1:200, outer = 50)
iis = repeat(1:50, inner = 200)
dk = DriscollKraay(Bartlett(5), tis = tis, iis = iis)
SUITE["LRV"]["DriscollKraay(Bartlett(5))"] = @benchmarkable aVar($dk, $X) seconds=SECONDS

# ---------------------------------------------------------------------------
# Regression models: 150 firms observed over 100 periods, sorted by firm, so the
# rows of a period cluster are not contiguous.
# ---------------------------------------------------------------------------
nfirm, nper = 150, 100
n = nfirm * nper
firm = repeat(1:nfirm, inner = nper)
period = repeat(1:nper, outer = nfirm)
df = DataFrame(ar1_matrix(rng, n, 5), :auto)
df.firm = firm
df.period = period
df.w = 0.5 .+ rand(rng, n)
df.y = df.x1 .- 0.5 .* df.x2 .+ randn(rng, nfirm)[firm] .+ randn(rng, n)
df.ybin = Float64.(df.y .> 0)
frm = @formula(y ~ x1 + x2 + x3 + x4 + x5)
ols = lm(frm, df)
wls = lm(frm, df, wts = df.w)
logit = glm(@formula(ybin ~ x1 + x2 + x3 + x4 + x5), df, Binomial())

SUITE["HC"] = BenchmarkGroup()
for (name, k) in ("HC0" => HC0(), "HC1" => HC1(), "HC3" => HC3())
    SUITE["HC"][name] = @benchmarkable vcov($k, $ols) seconds=SECONDS
end

SUITE["CR"] = BenchmarkGroup()
for (name, K) in ("CR0" => CR0, "CR1" => CR1, "CR2" => CR2, "CR3" => CR3)
    k = K(firm)
    SUITE["CR"]["$name firm"] = @benchmarkable vcov($k, $ols) seconds=SECONDS
end
for (name, K) in ("CR2" => CR2, "CR3" => CR3)
    k1 = K(period)
    k2 = K((firm, period))
    kf = K(firm)
    SUITE["CR"]["$name period"] = @benchmarkable vcov($k1, $ols) seconds=SECONDS
    SUITE["CR"]["$name firm+period"] = @benchmarkable vcov($k2, $ols) seconds=SECONDS
    SUITE["CR"]["$name firm weighted"] = @benchmarkable vcov($kf, $wls) seconds=SECONDS
    SUITE["CR"]["$name firm logit"] = @benchmarkable vcov($kf, $logit) seconds=SECONDS
end
cr2 = CR2(firm)
cached_cr2 = CachedCR(cr2, ols)
SUITE["CR"]["CachedCR(CR2) build"] = @benchmarkable CachedCR($cr2, $ols) seconds=SECONDS
SUITE["CR"]["CachedCR(CR2) vcov"] = @benchmarkable vcov($cached_cr2, $ols) seconds=SECONDS
cr0 = CR0(firm)
Xn = ar1_matrix(rng, n, 6)
SUITE["CR"]["CR0 firm matrix"] = @benchmarkable aVar($cr0, $Xn) seconds=SECONDS

# Small model: 50 clusters of 5 observations.
small = DataFrame(ar1_matrix(rng, 250, 5), :auto)
small.cl = repeat(1:50, inner = 5)
small.y = small.x1 .+ randn(rng, 250)
ols_small = lm(frm, small)
for (name, K) in ("CR0" => CR0, "CR2" => CR2, "CR3" => CR3)
    k = K(small.cl)
    SUITE["CR"]["$name small"] = @benchmarkable vcov($k, $ols_small) seconds=SECONDS
end

# ---------------------------------------------------------------------------
# Moment smoothing
# ---------------------------------------------------------------------------
SUITE["Smoothing"] = BenchmarkGroup()
for (T, k, m) in ((1_000, 5, 5), (10_000, 5, 5), (10_000, 20, 10))
    G = randn(rng, T, k)
    dest = similar(G)
    label = "T=$(T)_k=$(k)_m=$(m)"
    for (name, s) in ("Uniform" => UniformSmoother(m), "Triangular" =>
        TriangularSmoother(m))
        SUITE["Smoothing"]["$name $label"] = @benchmarkable CovarianceMatrices.smooth_moments(
            $G, $s) seconds=SECONDS
        SUITE["Smoothing"]["$name in-place $label"] = @benchmarkable CovarianceMatrices.smooth_moments!(
            $dest, $G, $s) seconds=SECONDS
    end
end
