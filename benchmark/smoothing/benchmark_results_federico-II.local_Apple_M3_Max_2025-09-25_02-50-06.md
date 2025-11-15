# 💻 System Information
Generated on: 2025-09-25T02:50:06.818

## 🚀 Julia & OS
| Characteristic  | Value |
|:----------------|:------|
| Julia Version   | 1.11.7 |
| OS              | Darwin (aarch64) |
| CPU Threads     | 10 |
| Hostname        | federico-II.local |
| User            | gragusa |

## 🧠 CPU
| Characteristic  | Value |
|:----------------|:------|
| Model           | Apple M3 Max |
| Speed           | 2400 MHz |

## 💾 Memory
| Type        | Size |
|:------------|:-----|
| Total RAM   | 96.00 GiB |
| Free RAM    | 4.46 GiB |


# 🚀 Comprehensive smooth_moments! Multi-Threading Benchmark

**Configuration:**
- Problem sizes: T ∈ {100, 500, 1000, 10000}
- Matrix dimensions: k = 5
- Bandwidth: 5.0 (UniformKernel)
- Threaded version: Activated for T > 800

## Performance Comparison

### Single-Threaded Methods (1 Thread Only)

| T | Single-arg In-place | Two-arg | Out-of-place |
|:---|:-------------------|:--------|:-------------|
| 100 | 2.1 μs | 2.3 μs | 2.4 μs |
| 500 | 9.7 μs | 10.2 μs | 10.4 μs |
| 1000 | 19.0 μs | 19.5 μs | 19.7 μs |
| 10000 | 190.2 μs | 191.3 μs | 191.3 μs |

### Threaded Method Performance

| T / Threads | 2t | 4t | 6t | 8t |
|:---|:---|:---|:---|:---|
| 100 | 20.2 μs | 23.2 μs | 30.8 μs | 40.1 μs |
| 500 | 34.6 μs | 30.9 μs | 36.5 μs | 42.4 μs |
| 1000 | 56.9 μs | 41.3 μs | 38.9 μs | 46.6 μs |
| 10000 | 270.2 μs | 148.4 μs | 126.9 μs | 124.4 μs |

## Threading Scaling Analysis

### T = 100

| Threads | Single-arg | Two-arg | Out-of-place | Threaded | Best Method |
|:--------|:-----------|:--------|:-------------|:---------|:------------|
| 1 | 2.1 μs | 2.3 μs | 2.4 μs | N/A | Single-arg |
| 2 | N/A | N/A | N/A | 20.2 μs | Threaded |
| 4 | N/A | N/A | N/A | 23.2 μs | Threaded |
| 6 | N/A | N/A | N/A | 30.8 μs | Threaded |
| 8 | N/A | N/A | N/A | 40.1 μs | Threaded |

### T = 500

| Threads | Single-arg | Two-arg | Out-of-place | Threaded | Best Method |
|:--------|:-----------|:--------|:-------------|:---------|:------------|
| 1 | 9.7 μs | 10.2 μs | 10.4 μs | N/A | Single-arg |
| 2 | N/A | N/A | N/A | 34.6 μs | Threaded |
| 4 | N/A | N/A | N/A | 30.9 μs | Threaded |
| 6 | N/A | N/A | N/A | 36.5 μs | Threaded |
| 8 | N/A | N/A | N/A | 42.4 μs | Threaded |

### T = 1000

| Threads | Single-arg | Two-arg | Out-of-place | Threaded | Best Method |
|:--------|:-----------|:--------|:-------------|:---------|:------------|
| 1 | 19.0 μs | 19.5 μs | 19.7 μs | N/A | Single-arg |
| 2 | N/A | N/A | N/A | 56.9 μs | Threaded |
| 4 | N/A | N/A | N/A | 41.3 μs | Threaded |
| 6 | N/A | N/A | N/A | 38.9 μs | Threaded |
| 8 | N/A | N/A | N/A | 46.6 μs | Threaded |

### T = 10000

| Threads | Single-arg | Two-arg | Out-of-place | Threaded | Best Method |
|:--------|:-----------|:--------|:-------------|:---------|:------------|
| 1 | 190.2 μs | 191.3 μs | 191.3 μs | N/A | Single-arg |
| 2 | N/A | N/A | N/A | 270.2 μs | Threaded |
| 4 | N/A | N/A | N/A | 148.4 μs | Threaded |
| 6 | N/A | N/A | N/A | 126.9 μs | Threaded |
| 8 | N/A | N/A | N/A | 124.4 μs | Threaded |

## Memory Allocation Analysis

### Single-Threaded Methods Memory Usage

| T | Single-arg In-place | Two-arg | Out-of-place |
|:---|:-------------------|:--------|:-------------|
| 100 | 1.8 KB | 992 B | 5.0 KB |
| 500 | 5.0 KB | 992 B | 21.0 KB |
| 1000 | 9.0 KB | 992 B | 65.0 KB |
| 10000 | 97.0 KB | 992 B | 417.0 KB |

### Threaded Method Memory Usage

| T / Threads | 2t | 4t | 6t | 8t |
|:---|:---|:---|:---|:---|
| 100 | 6.8 KB | 12.4 KB | 17.9 KB | 23.5 KB |
| 500 | 6.8 KB | 12.4 KB | 17.9 KB | 23.5 KB |
| 1000 | 6.8 KB | 12.4 KB | 17.9 KB | 23.5 KB |
| 10000 | 6.8 KB | 12.4 KB | 17.9 KB | 23.5 KB |

## Threading Efficiency Analysis

Speedup comparison: Threaded method vs best single-threaded method (Single-arg In-place)

### T = 100

| Threads | Threaded Time | Single-thread Baseline | Speedup | Efficiency |
|:--------|:-------------|:----------------------|:--------|:-----------|
| 2 | 20.2 μs | 2.1 μs | 9.7x slower | 5.2% |
| 4 | 23.2 μs | 2.1 μs | 11.12x slower | 2.2% |
| 6 | 30.8 μs | 2.1 μs | 14.78x slower | 1.1% |
| 8 | 40.1 μs | 2.1 μs | 19.26x slower | 0.6% |

### T = 500

| Threads | Threaded Time | Single-thread Baseline | Speedup | Efficiency |
|:--------|:-------------|:----------------------|:--------|:-----------|
| 2 | 34.6 μs | 9.7 μs | 3.58x slower | 14.0% |
| 4 | 30.9 μs | 9.7 μs | 3.2x slower | 7.8% |
| 6 | 36.5 μs | 9.7 μs | 3.78x slower | 4.4% |
| 8 | 42.4 μs | 9.7 μs | 4.39x slower | 2.8% |

### T = 1000

| Threads | Threaded Time | Single-thread Baseline | Speedup | Efficiency |
|:--------|:-------------|:----------------------|:--------|:-----------|
| 2 | 56.9 μs | 19.0 μs | 3.0x slower | 16.7% |
| 4 | 41.3 μs | 19.0 μs | 2.18x slower | 11.5% |
| 6 | 38.9 μs | 19.0 μs | 2.05x slower | 8.1% |
| 8 | 46.6 μs | 19.0 μs | 2.46x slower | 5.1% |

### T = 10000

| Threads | Threaded Time | Single-thread Baseline | Speedup | Efficiency |
|:--------|:-------------|:----------------------|:--------|:-----------|
| 2 | 270.2 μs | 190.2 μs | 1.42x slower | 35.2% |
| 4 | 148.4 μs | 190.2 μs | 1.28x faster | 32.0% |
| 6 | 126.9 μs | 190.2 μs | 1.5x faster | 25.0% |
| 8 | 124.4 μs | 190.2 μs | 1.53x faster | 19.1% |

## Performance Recommendations

### When to Use Threading

Based on the benchmark results:

- **T=100**: Single-threaded methods faster for all tested configurations
- **T=500**: Single-threaded methods faster for all tested configurations
- **T=1000**: Single-threaded methods faster for all tested configurations
- **T=10000**: Threading beneficial with 8+ threads (up to 1.53x speedup)

