using Random
using DataFrames
using Distributions
using RCall
using CategoricalArrays
using GLM
using CovarianceMatrices
using StableRNGs

function simulate_clustered_data(rng::AbstractRNG = Random.GLOBAL_RNG;
    num_clusters = 50,
    min_obs = 100,
    max_obs = 1000,
    β_0 = 2.0,
    β_1 = 1.5,
    σ_cluster = 2.0,  # Standard deviation of the cluster effect
    σ_error = 1.0     # Standard deviation of the idiosyncratic error
)
    # 1. Generate unbalanced cluster sizes
    # We pick a random number of observations for each of the 50 clusters
    cluster_sizes = rand(rng, min_obs:max_obs, num_clusters)
    total_obs = sum(cluster_sizes)

    # 2. Create the Cluster ID vector
    # We expand the cluster index 'i' by the size of that cluster
    # cl becomes: [1, 1, 1... 2, 2, 2... 50, 50]
    cl = vcat([fill(i, size) for (i, size) in enumerate(cluster_sizes)]...)

    # 3. Generate Regressor (X)
    x = rand(rng, Normal(0, 1), total_obs)

    # 4. Generate the Cluster Structure in Residuals

    # A: Cluster-specific Random Effect (u_c)
    # Generate 50 unique error terms (one per cluster)
    u_c_unique = rand(rng, Normal(0, σ_cluster), num_clusters)
    # Map these to the full length of the data based on cluster ID
    u_c_vector = u_c_unique[cl]

    # B: Idiosyncratic Error (ε_ic)
    # Generate unique error terms for every observation
    epsilon_ic = rand(rng, Normal(0, σ_error), total_obs)

    # C: Total Residual
    residuals = u_c_vector .+ epsilon_ic

    # 5. Calculate Y
    y = β_0 .+ (β_1 .* x) .+ residuals

    # 6. Assemble DataFrame
    df = DataFrame(
        cl = categorical(cl), # Treat cluster ID as a categorical factor
        x = x,
        y = y,
        true_residual = residuals,
        cluster_effect = u_c_vector,
        weights = rand(rng, Uniform(0.5, 1.5), total_obs)
    )

    return df
end

# Run the simulation

df = simulate_clustered_data(StableRNG(998877))

V_0a = [0.08531443236223397 -0.0012539581272725414; -0.0012539581272725414 0.0001474720536945924]
V_0b = [0.08705554322676935 -0.0012795491094617771; -0.001279549109461777 0.00015048168744346158]
V_1 =  [0.08705853770748022 -0.0012795931225709357; -0.0012795931225709357 0.00015048686361597961]
V_2 =  [0.08769414286741674 -0.001307347423372834; -0.001307347423372834 0.00015353402844790604]
V_3 =  [0.09014413753334793 -0.0013629470808218785; -0.0013629470808218785 0.00015987064688585098]

Vw_0a = [0.08516941951040422 -0.0010355677798112384; -0.0010355677798112382 0.00014685781078429842]
Vw_0b = [0.08690757092898391 -0.0010567018161339167; -0.0010567018161339167 0.00014985490896356988]
Vw_1 = [0.08691056031983863 -0.0010567381638848842; -0.0010567381638848842 0.0001498600635765639]
Vw_2 = [0.08772820788894604 -0.0010781705550907987; -0.0010781705550907987 0.00015271129710912336]
Vw_3 = [0.09037994446887648 -0.001122799656231728; -0.001122799656231728 0.00015889465356180393]




# The variance above are obtained from R's sandwich package
# @rput df
# R"""
# library(sandwich)
# lm_1 = lm(y ~ x, data = df)
# ## sandiwch seems to have peculiar way of dealing with HC0
# ## This is the unscaled variance
# V_0a = vcovCL(lm_1, cluster = df$cl, type = "HC0", cadjust = FALSE)
# ## This is, the default, the variance is actually multiplied by G/(G-1)
# V_0b = vcovCL(lm_1, cluster = df$cl, type = "HC0", cadjust = TRUE)
# V_1 = vcovCL(lm_1, cluster = df$cl, type = "HC1")
# V_2 = vcovCL(lm_1, cluster = df$cl, type = "HC2")
# V_3 = vcovCL(lm_1, cluster = df$cl, type = "HC3")

# lm_1w = lm(y ~ x, data = df, weights = weights)
# Vw_0a = vcovCL(lm_1w, cluster = df$cl, type = "HC0", cadjust = FALSE)
# ## This is, the default, the variance is actually multiplied by G/(G-1)
# Vw_0b = vcovCL(lm_1w, cluster = df$cl, type = "HC0", cadjust = TRUE)
# Vw_1 = vcovCL(lm_1w, cluster = df$cl, type = "HC1")
# Vw_2 = vcovCL(lm_1w, cluster = df$cl, type = "HC2")
# Vw_3 = vcovCL(lm_1w, cluster = df$cl, type = "HC3")
# """
# @rget V_0a V_0b V_1 V_2 V_3 Vw_0a Vw_0b Vw_1 Vw_2 Vw_3

lm_1 = lm(@formula(y ~ x), df)
V0 = vcov(CR0(df.cl), lm_1)
V1 = vcov(CR1(df.cl), lm_1)
V2 = vcov(CR2(df.cl), lm_1)
V3 = vcov(CR3(df.cl), lm_1)

lmw_1 = lm(@formula(y ~ x), df, wts = df.weights)
Vw0 = vcov(CR0(df.cl), lmw_1)
Vw1 = vcov(CR1(df.cl), lmw_1)
Vw2 = vcov(CR2(df.cl), lmw_1)
Vw3 = vcov(CR3(df.cl), lmw_1)

## This applies G/(G-k) correction internally
## So,
@test V0a ≈ V_0
n = nrow(df)
k = length(coef(lm_1))
g = length(levels(df.cl))
scale_back = ((n - 1) / (n - k) * (g / (g - 1)))
scale_new = (n-1)/(n-k)

@test V0 ≈ V_0a
@test V0 ≈ V_0b/(g/(g-1))
## Dealing with the different scaling conventions
## R: n-1 / n-k
## Julia: (n-1)/(n-k) * (G/(G-1))
@test (V1/scale_back)*scale_new ≈ V_1 rtol=1e-6
@test V2 ≈ V_2
@test V3 ≈ V_3


@test Vw0 ≈ Vw_0a
@test Vw0 ≈ Vw_0b/(g/(g-1))
## Dealing with the different scaling conventions
## R: n-1 / n-k
## Julia: (n-1)/(n-k) * (G/(G-1))
@test (Vw1/scale_back)*scale_new ≈ Vw_1 rtol=1e-6
@test Vw2 ≈ Vw_2 rtol=1e-2
@test Vw3 ≈ Vw_3 rtol=1e-2



