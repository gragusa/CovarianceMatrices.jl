# 🚀 Smoothed Moments Performance Benchmark Suite

This directory contains comprehensive benchmarking tools for the `smooth_moments!` implementations in CovarianceMatrices.jl.

## 📁 Files Overview

| File | Purpose |
|------|---------|
| `simple_multithreaded_benchmark.jl` | Core benchmark script that tests all smooth_moments implementations |
| `collect_results.jl` | Results collector and markdown report generator |
| `benchmark_results_*.md` | Machine-specific benchmark reports |
| `README.md` | This documentation |

## 🎯 What Gets Benchmarked

The benchmark suite tests **four different implementations** of smooth_moments:

1. **Single-argument In-place**: `smooth_moments!(G, weights, T)` - Modifies G directly
2. **Two-argument**: `smooth_moments!(result, G, weights, T)` - Separate result matrix
3. **Out-of-place**: `smooth_moments(G, weights, T)` - Returns new matrix
4. **Threaded**: `smooth_moments_threaded!(result, G, weights, T)` - Multi-threaded version (T > 800)

### Test Configuration

- **Problem sizes**: T ∈ {100, 500, 1000, 10000}
- **Matrix dimensions**: k = 5 columns
- **Bandwidth**: 5.0 (UniformSmoother)
- **Metrics**: Execution time, memory allocation, allocation count
- **Thread configurations**: 1, 2, 4, 6, 8 threads

## 🚀 How to Run the Benchmark

### Option 1: Quick Single-Thread Test

```bash
julia --project=. simple_multithreaded_benchmark.jl
```

### Option 2: Comprehensive Multi-Thread Analysis

Run the benchmark with different thread counts and collect results:

```bash
# Run benchmarks with different thread configurations
julia --project=. -t 1 simple_multithreaded_benchmark.jl > /tmp/benchmark_1t.out 2>&1
julia --project=. -t 2 simple_multithreaded_benchmark.jl > /tmp/benchmark_2t.out 2>&1
julia --project=. -t 4 simple_multithreaded_benchmark.jl > /tmp/benchmark_4t.out 2>&1
julia --project=. -t 6 simple_multithreaded_benchmark.jl > /tmp/benchmark_6t.out 2>&1
julia --project=. -t 8 simple_multithreaded_benchmark.jl > /tmp/benchmark_8t.out 2>&1

# Collect results and generate comprehensive report
julia collect_results.jl
```

### Option 3: Automated Script

Create a convenience script to run all configurations:

```bash
#!/bin/bash
# run_full_benchmark.sh

echo "🚀 Starting comprehensive smooth_moments benchmark..."
cd /path/to/CovarianceMatrices.jl/benchmark/smoothing

# Run benchmarks
for threads in 1 2 4 6 8; do
    echo "Running benchmark with $threads thread(s)..."
    julia --project=../.. -t $threads simple_multithreaded_benchmark.jl > /tmp/benchmark_${threads}t.out 2>&1
done

# Collect results
echo "📊 Collecting and analyzing results..."
julia collect_results.jl

echo "✅ Benchmark complete! Check the generated markdown file."
```

## 📊 Understanding the Results

The generated markdown report includes:

### 1. System Information
- Hardware specifications (CPU, memory, OS)
- Julia version and configuration
- Available CPU threads

### 2. Performance Comparison Tables
- **By Method**: Time comparison across thread counts for each implementation
- **By Problem Size**: Performance scaling analysis

### 3. Threading Analysis
- **Scaling Analysis**: How each method performs with different thread counts
- **Best Method**: Winner for each configuration
- **Threading Efficiency**: Speedup and efficiency metrics

### 4. Memory Analysis
- Memory allocation patterns
- Allocation count comparison
- Memory efficiency rankings

## 🏆 Key Insights from Benchmarks

Based on typical results (Apple M3 Max example):

### Performance Winners
- **Small problems (T ≤ 500)**: Single-argument in-place (~1-10 μs)
- **Medium problems (T = 1000)**: Single-argument in-place (~18 μs)
- **Large problems (T = 10000)**: Threaded version with 6-8 threads (~120 μs)

### Memory Efficiency
- **Most efficient**: Two-argument (constant ~1KB)
- **Moderate**: Single-argument in-place (scales with T)
- **Highest usage**: Out-of-place (scales significantly with T)

### Threading Recommendations
- **Threading overhead significant** for small-medium problems
- **Break-even point**: Around T = 5000-10000 with 4+ threads
- **Current auto-threading threshold (T > 800)** may be too aggressive

## 🔧 Customizing the Benchmark

### Changing Problem Sizes

Edit `simple_multithreaded_benchmark.jl`:
```julia
T_values = [100, 500, 1000, 10000]  # Modify these values
k = 5  # Matrix columns
bandwidth = 5.0  # Kernel bandwidth
```

### Adding Different Kernels

```julia
# Test both UniformSmoother and TriangularSmoother
kernels = [CovarianceMatrices.UniformSmoother(), CovarianceMatrices.TriangularSmoother()]
```

### Different Thread Configurations

Modify the collection script to test different thread counts:
```bash
for threads in 1 3 5 7 9 10; do
    julia --project=../.. -t $threads simple_multithreaded_benchmark.jl > /tmp/benchmark_${threads}t.out 2>&1
done
```

## 📈 Interpreting Results

### Good Performance Indicators
- **Low execution times** (μs range for typical problems)
- **Consistent performance** across runs
- **Effective threading** (speedup with more threads for large problems)
- **Low memory allocation** for memory-constrained scenarios

### Red Flags
- **Significant threading overhead** for small problems
- **Memory allocation scaling** faster than problem size
- **Inconsistent performance** across similar configurations

## 🐛 Troubleshooting

### Common Issues

1. **"Method not found" errors**: Ensure you're in the CovarianceMatrices.jl project directory
2. **Threading not working**: Check Julia was started with `-t N` flag
3. **Missing results**: Verify output files were created in `/tmp/`
4. **Performance variations**: Results can vary due to system load, thermal throttling

### Getting Help

- Check the main CovarianceMatrices.jl documentation
- Verify all dependencies are installed: `using Pkg; Pkg.test()`
- For threading issues, check: `Threads.nthreads()` in Julia

## 📝 Output Files

Generated files follow the naming convention:
```
benchmark_results_[hostname]_[CPU-model]_[timestamp].md
```

Example:
```
benchmark_results_federico-II.local_Apple_M3_Max_2025-09-25_00-28-40.md
```

This ensures benchmark results are:
- **Machine-specific**: Easy to compare across different systems
- **Timestamped**: Track performance over time
- **Self-documenting**: Include system specs in the report

Happy benchmarking! 🚀