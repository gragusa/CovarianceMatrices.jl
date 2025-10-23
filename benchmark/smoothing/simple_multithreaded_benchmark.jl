#!/usr/bin/env julia

"""
Simple multi-threaded benchmark for smooth_moments implementations
Run this script with different -t settings: julia -t 1, julia -t 2, etc.
"""

using CovarianceMatrices
using BenchmarkTools
using StableRNGs
using Printf
using Dates

# System info functions
function bytes_to_gib(bytes::Integer)
    gib = bytes / 1024^3
    return @sprintf("%.2f GiB", gib)
end

function get_system_info()
    buffer = IOBuffer()

    println(buffer, "# 💻 System Information")
    println(buffer, "Generated on: $(now())")

    println(buffer, "\n## 🚀 Julia & OS")
    println(buffer, "| Characteristic  | Value |")
    println(buffer, "|:----------------|:------|")
    @printf(buffer, "| Julia Version   | %s |\n", VERSION)
    @printf(buffer, "| OS              | %s (%s) |\n", Sys.KERNEL, Sys.ARCH)
    @printf(buffer, "| CPU Threads     | %d |\n", Sys.CPU_THREADS)
    @printf(buffer, "| Hostname        | %s |\n", gethostname())

    println(buffer, "\n## 🧠 CPU")
    println(buffer, "| Characteristic  | Value |")
    println(buffer, "|:----------------|:------|")
    cpu_info = Sys.cpu_info()
    @printf(buffer, "| Model           | %s |\n", cpu_info[1].model)
    @printf(buffer, "| Speed           | %d MHz |\n", cpu_info[1].speed)

    println(buffer, "\n## 💾 Memory")
    println(buffer, "| Type        | Size |")
    println(buffer, "|:------------|:-----|")
    @printf(buffer, "| Total RAM   | %s |\n", bytes_to_gib(Sys.total_memory()))
    @printf(buffer, "| Free RAM    | %s |\n", bytes_to_gib(Sys.free_memory()))

    return String(take!(buffer))
end

function format_memory(bytes)
    if isnan(bytes)
        return "N/A"
    elseif bytes < 1024
        return "$(round(Int, bytes)) B"
    elseif bytes < 1024^2
        return "$(round(bytes/1024, digits=1)) KB"
    elseif bytes < 1024^3
        return "$(round(bytes/1024^2, digits=1)) MB"
    else
        return "$(round(bytes/1024^3, digits=1)) GB"
    end
end

function format_time(ms)
    if isnan(ms)
        return "N/A"
    elseif ms < 1.0
        return "$(round(ms*1000, digits=1)) μs"
    elseif ms < 1000.0
        return "$(round(ms, digits=2)) ms"
    else
        return "$(round(ms/1000, digits=2)) s"
    end
end

function run_benchmark()
    println("🚀 smooth_moments! Benchmark with $(Threads.nthreads()) thread(s)")
    println("=" ^ 60)

    # Configuration
    T_values = [100, 500, 1000, 10000]
    k = 5
    bandwidth = 5.0
    kernel = CovarianceMatrices.UniformSmoother()
    rng = StableRNG(42)

    # Results storage
    results = []
    current_threads = Threads.nthreads()

    for T in T_values
        println("📊 Benchmarking T=$T...")

        # Generate test data
        X = randn(rng, T, k)
        weights = CovarianceMatrices.compute_weights(kernel, bandwidth, T, Float64)

        # Method 1: Single-argument in-place (only benchmark on 1 thread)
        time1, allocs1,
        memory1 = if current_threads == 1
            X_test = copy(X)
            bench1 = @benchmark CovarianceMatrices.smooth_moments!($X_test, $weights, $T) setup=(X_test = copy($X))
            (minimum(bench1.times) / 1e6, bench1.allocs, bench1.memory)
        else
            (NaN, NaN, NaN)  # Skip for multi-threaded runs
        end

        # Method 2: Two-argument (only benchmark on 1 thread)
        time2, allocs2,
        memory2 = if current_threads == 1
            X_test = copy(X)
            result = similar(X)
            bench2 = @benchmark CovarianceMatrices.smooth_moments!($result, $X_test, $weights, $T) setup=(
                X_test = copy($X); result = similar($X))
            (minimum(bench2.times) / 1e6, bench2.allocs, bench2.memory)
        else
            (NaN, NaN, NaN)  # Skip for multi-threaded runs
        end

        # Method 3: Out-of-place (only benchmark on 1 thread)
        time3, allocs3,
        memory3 = if current_threads == 1
            X_test = copy(X)
            bench3 = @benchmark CovarianceMatrices.smooth_moments($X_test, $weights, $T) setup=(X_test = copy($X))
            (minimum(bench3.times) / 1e6, bench3.allocs, bench3.memory)
        else
            (NaN, NaN, NaN)  # Skip for multi-threaded runs
        end

        # Method 4: Threaded (always benchmark if threads > 1)
        time4, allocs4,
        memory4 = if current_threads > 1
            X_test = copy(X)
            result_threaded = similar(X)
            bench4 = @benchmark CovarianceMatrices.smooth_moments_threaded!(
                $result_threaded, $X_test, $weights, $T) setup=(
                X_test = copy($X); result_threaded = similar($X))
            (minimum(bench4.times) / 1e6, bench4.allocs, bench4.memory)
        else
            (NaN, NaN, NaN)  # Skip for single-threaded runs
        end

        push!(results,
            (
                T = T,
                threads = current_threads,
                single_time = time1, single_allocs = allocs1, single_memory = memory1,
                two_arg_time = time2, two_arg_allocs = allocs2, two_arg_memory = memory2,
                out_of_place_time = time3, out_of_place_allocs = allocs3, out_of_place_memory = memory3,
                threaded_time = time4, threaded_allocs = allocs4, threaded_memory = memory4
            ))
    end

    return results
end

# Run benchmark
results = run_benchmark()

# Print results in a format that can be collected
println("\n" * "="^60)
println("RESULTS_START")
for r in results
    println("$(r.threads),$(r.T),$(r.single_time),$(r.single_allocs),$(r.single_memory),$(r.two_arg_time),$(r.two_arg_allocs),$(r.two_arg_memory),$(r.out_of_place_time),$(r.out_of_place_allocs),$(r.out_of_place_memory),$(r.threaded_time),$(r.threaded_allocs),$(r.threaded_memory)")
end
println("RESULTS_END")

# Also print human-readable summary
println("\n📊 Results for $(results[1].threads) thread(s):")
println("| T | Single-arg | Two-arg | Out-of-place | Threaded |")
println("|---|-----------|---------|--------------|----------|")
for r in results
    println("| $(r.T) | $(format_time(r.single_time)) | $(format_time(r.two_arg_time)) | $(format_time(r.out_of_place_time)) | $(format_time(r.threaded_time)) |")
end
