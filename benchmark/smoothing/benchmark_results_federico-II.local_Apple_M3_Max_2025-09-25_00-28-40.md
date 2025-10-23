# 💻 System Information
Generated on: 2025-09-25T00:28:40.540

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
| Free RAM    | 7.90 GiB |


# 🚀 Comprehensive smooth_moments! Multi-Threading Benchmark

**Configuration:**
- Problem sizes: T ∈ {100, 500, 1000, 10000}
- Matrix dimensions: k = 5
- Bandwidth: 5.0 (UniformSmoother)
- Threaded version: Activated for T > 800

## Performance Comparison by Method

### Single-arg In-place Method

| T / Threads | 1t | 2t | 4t | 6t | 8t |
|:---|:---|:---|:---|:---|:---|
| 100 | 1.9 μs | 2.1 μs | 2.1 μs | 2.1 μs | 2.1 μs |
| 500 | 8.9 μs | 9.5 μs | 9.6 μs | 9.5 μs | 9.5 μs |
| 1000 | 17.7 μs | 18.8 μs | 18.9 μs | 17.8 μs | 18.9 μs |
| 10000 | 178.1 μs | 189.2 μs | 189.3 μs | 183.8 μs | 189.4 μs |

### Two-arg Method

| T / Threads | 1t | 2t | 4t | 6t | 8t |
|:---|:---|:---|:---|:---|:---|
| 100 | 2.2 μs | 2.3 μs | 2.3 μs | 2.2 μs | 2.2 μs |
| 500 | 9.6 μs | 10.0 μs | 10.0 μs | 9.4 μs | 10.0 μs |
| 1000 | 18.7 μs | 19.6 μs | 19.5 μs | 18.5 μs | 19.6 μs |
| 10000 | 182.6 μs | 182.8 μs | 187.1 μs | 178.5 μs | 188.0 μs |

### Out-of-place Method

| T / Threads | 1t | 2t | 4t | 6t | 8t |
|:---|:---|:---|:---|:---|:---|
| 100 | 2.2 μs | 2.2 μs | 2.4 μs | 2.4 μs | 2.4 μs |
| 500 | 9.9 μs | 10.2 μs | 10.1 μs | 10.2 μs | 10.2 μs |
| 1000 | 18.5 μs | 19.6 μs | 19.7 μs | 19.7 μs | 19.7 μs |
| 10000 | 182.4 μs | 182.7 μs | 191.1 μs | 181.8 μs | 189.1 μs |

### Threaded Method

| T / Threads | 1t | 2t | 4t | 6t | 8t |
|:---|:---|:---|:---|:---|:---|
| 100 | N/A | N/A | N/A | N/A | N/A |
| 500 | N/A | N/A | N/A | N/A | N/A |
| 1000 | N/A | 56.5 μs | 41.4 μs | 39.5 μs | 45.6 μs |
| 10000 | N/A | 265.1 μs | 148.6 μs | 124.6 μs | 121.4 μs |

## Threading Scaling Analysis

### T = 100

| Threads | Single-arg | Two-arg | Out-of-place | Threaded | Best Method |
|:--------|:-----------|:--------|:-------------|:---------|:------------|
| 1 | 1.9 μs | 2.2 μs | 2.2 μs | N/A | Single-arg |
| 2 | 2.1 μs | 2.3 μs | 2.2 μs | N/A | Single-arg |
| 4 | 2.1 μs | 2.3 μs | 2.4 μs | N/A | Single-arg |
| 6 | 2.1 μs | 2.2 μs | 2.4 μs | N/A | Single-arg |
| 8 | 2.1 μs | 2.2 μs | 2.4 μs | N/A | Single-arg |

### T = 500

| Threads | Single-arg | Two-arg | Out-of-place | Threaded | Best Method |
|:--------|:-----------|:--------|:-------------|:---------|:------------|
| 1 | 8.9 μs | 9.6 μs | 9.9 μs | N/A | Single-arg |
| 2 | 9.5 μs | 10.0 μs | 10.2 μs | N/A | Single-arg |
| 4 | 9.6 μs | 10.0 μs | 10.1 μs | N/A | Single-arg |
| 6 | 9.5 μs | 9.4 μs | 10.2 μs | N/A | Two-arg |
| 8 | 9.5 μs | 10.0 μs | 10.2 μs | N/A | Single-arg |

### T = 1000

| Threads | Single-arg | Two-arg | Out-of-place | Threaded | Best Method |
|:--------|:-----------|:--------|:-------------|:---------|:------------|
| 1 | 17.7 μs | 18.7 μs | 18.5 μs | N/A | Single-arg |
| 2 | 18.8 μs | 19.6 μs | 19.6 μs | 56.5 μs | Single-arg |
| 4 | 18.9 μs | 19.5 μs | 19.7 μs | 41.4 μs | Single-arg |
| 6 | 17.8 μs | 18.5 μs | 19.7 μs | 39.5 μs | Single-arg |
| 8 | 18.9 μs | 19.6 μs | 19.7 μs | 45.6 μs | Single-arg |

### T = 10000

| Threads | Single-arg | Two-arg | Out-of-place | Threaded | Best Method |
|:--------|:-----------|:--------|:-------------|:---------|:------------|
| 1 | 178.1 μs | 182.6 μs | 182.4 μs | N/A | Single-arg |
| 2 | 189.2 μs | 182.8 μs | 182.7 μs | 265.1 μs | Out-of-place |
| 4 | 189.3 μs | 187.1 μs | 191.1 μs | 148.6 μs | Threaded |
| 6 | 183.8 μs | 178.5 μs | 181.8 μs | 124.6 μs | Threaded |
| 8 | 189.4 μs | 188.0 μs | 189.1 μs | 121.4 μs | Threaded |

## Memory Allocation Analysis

Memory allocation is consistent across thread counts:

| Method | Typical Memory Usage |
|:-------|:---------------------|
| Single-arg In-place | 1.8 KB |
| Two-arg | 992 B |
| Out-of-place | 5.0 KB |

## Threading Efficiency

Speedup comparison (compared to 1 thread):

### T = 1000 (Threaded Method)

| Threads | Time | Speedup vs 1t | Efficiency |
|:--------|:-----|:--------------|:-----------|
| 1 | 17.7 μs | 1.0x | 100.0% |
| 2 | 56.5 μs | 0.31x | 15.7% |
| 4 | 41.4 μs | 0.43x | 10.7% |
| 6 | 39.5 μs | 0.45x | 7.5% |
| 8 | 45.6 μs | 0.39x | 4.9% |

### T = 10000 (Threaded Method)

| Threads | Time | Speedup vs 1t | Efficiency |
|:--------|:-----|:--------------|:-----------|
| 1 | 178.1 μs | 1.0x | 100.0% |
| 2 | 265.1 μs | 0.67x | 33.6% |
| 4 | 148.6 μs | 1.2x | 30.0% |
| 6 | 124.6 μs | 1.43x | 23.8% |
| 8 | 121.4 μs | 1.47x | 18.3% |

