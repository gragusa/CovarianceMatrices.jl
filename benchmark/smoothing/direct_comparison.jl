using BenchmarkTools
using Random
using CovarianceMatrices

# Set up test parameters
Random.seed!(123)
T = 1000
k = 5
bandwidth = 30.0

# Generate test data
G = randn(T, k)
G_copy1 = copy(G)
G_copy2 = copy(G)

# Create kernel for new implementation
kernel = CovarianceMatrices.UniformSmoother()

println("🔬 Direct Algorithm Comparison")
println("=" ^ 50)
println("Parameters: T=$T, k=$k, bandwidth=$bandwidth")
println()

# Benchmark new kernel-based implementation
println("🆕 NEW: Kernel-based implementation")
result_new = @benchmark CovarianceMatrices.smooth_moments!($G_copy1, $kernel, $bandwidth, $T) setup=(copyto!(
    $G_copy1, $G))
display(result_new)
println()

# Benchmark old weight-based implementation
println("🗂️  OLD: Weight-based implementation")
weights = CovarianceMatrices.compute_weights(kernel, bandwidth, T, Float64)
result_old = @benchmark CovarianceMatrices.smooth_moments!($G_copy2, $weights, $T) setup=(copyto!($G_copy2, $G))
display(result_old)
println()

# Show memory comparison
println("💾 Memory Usage Comparison:")
println("NEW (kernel-based): $(BenchmarkTools.memory(result_new)) bytes")
println("OLD (weight-based): $(BenchmarkTools.memory(result_old)) bytes")
println("Memory reduction: $(round((1 - BenchmarkTools.memory(result_new)/BenchmarkTools.memory(result_old))*100, digits=1))%")
println()

# Show performance comparison
time_new = BenchmarkTools.median(result_new).time
time_old = BenchmarkTools.median(result_old).time
speedup = time_old / time_new

println("⚡ Performance Comparison:")
println("NEW median time: $(round(time_new/1000, digits=1)) μs")
println("OLD median time: $(round(time_old/1000, digits=1)) μs")
if speedup > 1
    println("Speedup: $(round(speedup, digits=2))x faster")
else
    println("Slowdown: $(round(1/speedup, digits=2))x slower")
end
println()

# Verify results are identical
copyto!(G_copy1, G)
copyto!(G_copy2, G)
CovarianceMatrices.smooth_moments!(G_copy1, kernel, bandwidth, T)
CovarianceMatrices.smooth_moments!(G_copy2, weights, T)

max_diff = maximum(abs.(G_copy1 - G_copy2))
println("🔍 Correctness Check:")
println("Maximum difference between algorithms: $(max_diff)")
println("Results identical: $(max_diff < 1e-14)")
