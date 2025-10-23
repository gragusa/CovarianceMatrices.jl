#!/usr/bin/env julia

using Dates
using Printf

# Function definitions from the benchmark script
function bytes_to_gib(bytes::Integer)
    gib = bytes / 1024^3
    return @sprintf("%.2f GiB", gib)
end

function get_system_info()
    buffer = IOBuffer()

    println(buffer, "# 💻 System Information")
    println(buffer, "Generated on: $(now())")
    println(buffer, "")

    println(buffer, "## 🚀 Julia & OS")
    println(buffer, "| Characteristic  | Value |")
    println(buffer, "|:----------------|:------|")
    @printf(buffer, "| Julia Version   | %s |\n", VERSION)
    @printf(buffer, "| OS              | %s (%s) |\n", Sys.KERNEL, Sys.ARCH)
    @printf(buffer, "| CPU Threads     | %d |\n", Sys.CPU_THREADS)
    @printf(buffer, "| Hostname        | %s |\n", gethostname())
    @printf(buffer, "| User            | %s |\n",
        get(ENV, "USER", get(ENV, "USERNAME", "N/A")))
    println(buffer, "")

    println(buffer, "## 🧠 CPU")
    println(buffer, "| Characteristic  | Value |")
    println(buffer, "|:----------------|:------|")
    cpu_info = Sys.cpu_info()
    @printf(buffer, "| Model           | %s |\n", cpu_info[1].model)
    @printf(buffer, "| Speed           | %d MHz |\n", cpu_info[1].speed)
    println(buffer, "")

    println(buffer, "## 💾 Memory")
    println(buffer, "| Type        | Size |")
    println(buffer, "|:------------|:-----|")
    @printf(buffer, "| Total RAM   | %s |\n", bytes_to_gib(Sys.total_memory()))
    @printf(buffer, "| Free RAM    | %s |\n", bytes_to_gib(Sys.free_memory()))
    println(buffer, "")

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

function parse_results(filename)
    results = []
    if isfile(filename)
        content = read(filename, String)
        lines = split(content, '\n')

        in_results = false
        for line in lines
            if startswith(line, "RESULTS_START")
                in_results = true
                continue
            elseif startswith(line, "RESULTS_END")
                break
            elseif in_results && !isempty(line)
                parts = split(line, ',')
                if length(parts) >= 14
                    threads = parse(Int, parts[1])
                    T = parse(Int, parts[2])
                    single_time = parts[3] == "NaN" ? NaN : parse(Float64, parts[3])
                    single_allocs = parts[4] == "NaN" ? NaN : parse(Float64, parts[4])
                    single_memory = parts[5] == "NaN" ? NaN : parse(Float64, parts[5])
                    two_arg_time = parts[6] == "NaN" ? NaN : parse(Float64, parts[6])
                    two_arg_allocs = parts[7] == "NaN" ? NaN : parse(Float64, parts[7])
                    two_arg_memory = parts[8] == "NaN" ? NaN : parse(Float64, parts[8])
                    out_of_place_time = parts[9] == "NaN" ? NaN : parse(Float64, parts[9])
                    out_of_place_allocs = parts[10] == "NaN" ? NaN :
                                          parse(Float64, parts[10])
                    out_of_place_memory = parts[11] == "NaN" ? NaN :
                                          parse(Float64, parts[11])

                    threaded_time = parts[12] == "NaN" ? NaN : parse(Float64, parts[12])
                    threaded_allocs = parts[13] == "NaN" ? NaN : parse(Float64, parts[13])
                    threaded_memory = parts[14] == "NaN" ? NaN : parse(Float64, parts[14])

                    push!(results,
                        (
                            threads = threads, T = T,
                            single_time = single_time, single_allocs = single_allocs, single_memory = single_memory,
                            two_arg_time = two_arg_time, two_arg_allocs = two_arg_allocs,
                            two_arg_memory = two_arg_memory,
                            out_of_place_time = out_of_place_time, out_of_place_allocs = out_of_place_allocs,
                            out_of_place_memory = out_of_place_memory,
                            threaded_time = threaded_time, threaded_allocs = threaded_allocs,
                            threaded_memory = threaded_memory
                        ))
                end
            end
        end
    end
    return results
end

function generate_markdown_report(all_results)
    report = IOBuffer()

    # System information
    println(report, get_system_info())

    println(report, "# 🚀 Comprehensive smooth_moments! Multi-Threading Benchmark")
    println(report, "")
    println(report, "**Configuration:**")
    println(report, "- Problem sizes: T ∈ {100, 500, 1000, 10000}")
    println(report, "- Matrix dimensions: k = 5")
    println(report, "- Bandwidth: 5.0 (UniformSmoother)")
    println(report, "- Threaded version: Activated for T > 800")
    println(report, "")

    # Get unique values
    T_values = sort(unique([r.T for r in all_results]))
    thread_counts = sort(unique([r.threads for r in all_results]))

    println(report, "## Performance Comparison")
    println(report, "")

    # Separate tables for non-threaded and threaded methods

    # Non-threaded methods (only have data for 1 thread)
    println(report, "### Single-Threaded Methods (1 Thread Only)")
    println(report, "")
    println(report, "| T | Single-arg In-place | Two-arg | Out-of-place |")
    println(report, "|:---|:-------------------|:--------|:-------------|")

    # Find results with 1 thread
    single_thread_results = filter(r -> r.threads == 1, all_results)
    for T in T_values
        result = findfirst(r -> r.T == T, single_thread_results)
        if result !== nothing
            r = single_thread_results[result]
            println(report,
                "| $(T) | $(format_time(r.single_time)) | $(format_time(r.two_arg_time)) | $(format_time(r.out_of_place_time)) |")
        end
    end
    println(report, "")

    # Threaded method performance across different thread counts
    println(report, "### Threaded Method Performance")
    println(report, "")
    println(report, "| T / Threads | ", join(["$(t)t" for t in thread_counts if t > 1], " | "), " |")
    println(report, "|",
        join([":---" for _ in 1:(length([t for t in thread_counts if t > 1]) + 1)], "|"),
        "|")

    for T in T_values
        times_row = String[]
        for threads in thread_counts
            if threads > 1  # Only show threaded results
                result = findfirst(r -> r.T == T && r.threads == threads, all_results)
                if result !== nothing
                    time_val = all_results[result].threaded_time
                    push!(times_row, format_time(time_val))
                else
                    push!(times_row, "N/A")
                end
            end
        end
        if !isempty(times_row)
            println(report, "| $(T) | ", join(times_row, " | "), " |")
        end
    end
    println(report, "")

    # Threading scaling analysis
    println(report, "## Threading Scaling Analysis")
    println(report, "")

    for T in T_values
        T_results = filter(r -> r.T == T, all_results)
        if length(T_results) > 1
            println(report, "### T = $(T)")
            println(report, "")
            println(report, "| Threads | Single-arg | Two-arg | Out-of-place | Threaded | Best Method |")
            println(report, "|:--------|:-----------|:--------|:-------------|:---------|:------------|")

            for threads in thread_counts
                result_idx = findfirst(r -> r.T == T && r.threads == threads, T_results)
                if result_idx !== nothing
                    r = T_results[result_idx]
                    times = [
                        ("Single-arg", r.single_time),
                        ("Two-arg", r.two_arg_time),
                        ("Out-of-place", r.out_of_place_time),
                        ("Threaded", r.threaded_time)
                    ]
                    # Filter out NaN results
                    valid_times = filter(t -> !isnan(t[2]), times)
                    best_method = if !isempty(valid_times)
                        min_idx = argmin([t[2] for t in valid_times])
                        valid_times[min_idx][1]
                    else
                        "N/A"
                    end

                    println(report,
                        "| $(threads) | $(format_time(r.single_time)) | $(format_time(r.two_arg_time)) | $(format_time(r.out_of_place_time)) | $(format_time(r.threaded_time)) | $(best_method) |")
                end
            end
            println(report, "")
        end
    end

    # Memory analysis
    println(report, "## Memory Allocation Analysis")
    println(report, "")

    # Show memory usage for non-threaded methods (from 1-thread run)
    single_thread_result = findfirst(r -> r.threads == 1, all_results)
    if single_thread_result !== nothing
        println(report, "### Single-Threaded Methods Memory Usage")
        println(report, "")
        println(report, "| T | Single-arg In-place | Two-arg | Out-of-place |")
        println(report, "|:---|:-------------------|:--------|:-------------|")

        for T in T_values
            result = findfirst(r -> r.T == T && r.threads == 1, all_results)
            if result !== nothing
                r = all_results[result]
                println(report,
                    "| $(T) | $(format_memory(r.single_memory)) | $(format_memory(r.two_arg_memory)) | $(format_memory(r.out_of_place_memory)) |")
            end
        end
        println(report, "")
    end

    # Show memory usage for threaded method across different thread counts
    println(report, "### Threaded Method Memory Usage")
    println(report, "")
    println(report, "| T / Threads | ", join(["$(t)t" for t in thread_counts if t > 1], " | "), " |")
    println(report, "|",
        join([":---" for _ in 1:(length([t for t in thread_counts if t > 1]) + 1)], "|"),
        "|")

    for T in T_values
        memory_row = String[]
        for threads in thread_counts
            if threads > 1
                result = findfirst(r -> r.T == T && r.threads == threads, all_results)
                if result !== nothing
                    memory_val = all_results[result].threaded_memory
                    push!(memory_row, format_memory(memory_val))
                else
                    push!(memory_row, "N/A")
                end
            end
        end
        if !isempty(memory_row)
            println(report, "| $(T) | ", join(memory_row, " | "), " |")
        end
    end
    println(report, "")

    # Threading efficiency analysis
    println(report, "## Threading Efficiency Analysis")
    println(report, "")
    println(report,
        "Speedup comparison: Threaded method vs best single-threaded method (Single-arg In-place)")
    println(report, "")

    for T in T_values
        println(report, "### T = $(T)")
        println(report, "")
        println(report, "| Threads | Threaded Time | Single-thread Baseline | Speedup | Efficiency |")
        println(report, "|:--------|:-------------|:----------------------|:--------|:-----------|")

        # Get single-threaded baseline (best performing single-threaded method)
        baseline_result = findfirst(r -> r.T == T && r.threads == 1, all_results)
        baseline_time = if baseline_result !== nothing
            all_results[baseline_result].single_time  # Use single-arg as baseline
        else
            NaN
        end

        if !isnan(baseline_time)
            for threads in thread_counts
                if threads > 1  # Only compare multi-threaded results
                    result_idx = findfirst(r -> r.T == T && r.threads == threads, all_results)
                    if result_idx !== nothing
                        r = all_results[result_idx]
                        if !isnan(r.threaded_time)
                            speedup = baseline_time / r.threaded_time
                            efficiency = speedup / threads * 100

                            speedup_str = if speedup > 1
                                "$(round(speedup, digits=2))x faster"
                            else
                                "$(round(1/speedup, digits=2))x slower"
                            end

                            println(report,
                                "| $(threads) | $(format_time(r.threaded_time)) | $(format_time(baseline_time)) | $(speedup_str) | $(round(efficiency, digits=1))% |")
                        end
                    end
                end
            end
        end
        println(report, "")
    end

    # Summary recommendations
    println(report, "## Performance Recommendations")
    println(report, "")

    # Find break-even points
    println(report, "### When to Use Threading")
    println(report, "")
    println(report, "Based on the benchmark results:")
    println(report, "")

    for T in T_values
        baseline_result = findfirst(r -> r.T == T && r.threads == 1, all_results)
        if baseline_result !== nothing
            baseline_time = all_results[baseline_result].single_time

            # Find the first thread count where threaded is faster
            faster_configs = []
            for threads in thread_counts
                if threads > 1
                    result_idx = findfirst(r -> r.T == T && r.threads == threads, all_results)
                    if result_idx !== nothing
                        r = all_results[result_idx]
                        if !isnan(r.threaded_time) && r.threaded_time < baseline_time
                            speedup = baseline_time / r.threaded_time
                            push!(faster_configs, (threads, speedup))
                        end
                    end
                end
            end

            if !isempty(faster_configs)
                best_idx = argmax([x[2] for x in faster_configs])
                best_config = faster_configs[best_idx]
                println(report,
                    "- **T=$(T)**: Threading beneficial with $(best_config[1])+ threads (up to $(round(best_config[2], digits=2))x speedup)")
            else
                println(report,
                    "- **T=$(T)**: Single-threaded methods faster for all tested configurations")
            end
        end
    end

    println(report, "")

    return String(take!(report))
end

function get_machine_identifier()
    hostname = gethostname()
    cpu_model = Sys.cpu_info()[1].model
    # Create a safe filename from CPU model
    cpu_safe = replace(cpu_model, r"[^a-zA-Z0-9_-]" => "_")
    cpu_safe = replace(cpu_safe, r"_+" => "_")  # Replace multiple underscores with single
    cpu_safe = strip(cpu_safe, '_')  # Remove leading/trailing underscores

    return "$(hostname)_$(cpu_safe)"
end

# Main execution
function main()
    println("📊 Collecting benchmark results...")

    # Collect results from all thread configurations
    all_results = []
    thread_files = [
        ("/tmp/benchmark_1t.out", 1),
        ("/tmp/benchmark_2t.out", 2),
        ("/tmp/benchmark_4t.out", 4),
        ("/tmp/benchmark_6t.out", 6),
        ("/tmp/benchmark_8t.out", 8)
    ]

    for (filename, expected_threads) in thread_files
        results = parse_results(filename)
        if !isempty(results)
            append!(all_results, results)
            println("✅ Loaded results from $(length(results)) benchmarks with $(expected_threads) threads")
        else
            println("⚠️  No results found in $(filename)")
        end
    end

    if isempty(all_results)
        println("❌ No results collected!")
        return
    end

    println("📈 Total results collected: $(length(all_results))")

    # Generate machine-specific report
    machine_id = get_machine_identifier()
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    filename = "benchmark_results_$(machine_id)_$(timestamp).md"

    println("📝 Generating comprehensive report...")
    report = generate_markdown_report(all_results)

    # Save report in current directory
    filepath = joinpath(@__DIR__, filename)
    open(filepath, "w") do f
        write(f, report)
    end

    println("✅ Report saved to: $(filepath)")
    println("")
    println("📊 Summary:")
    println("- Benchmarked $(length(unique([r.threads for r in all_results]))) thread configurations")
    println("- Tested $(length(unique([r.T for r in all_results]))) problem sizes")
    println("- Generated $(length(all_results)) total data points")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
