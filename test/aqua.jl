## Aqua.jl automated quality assurance tests
##
## This file runs a comprehensive suite of QA checks including:
## - Method ambiguities detection
## - Stale dependency checks
## - Invalid exports validation
## - Project.toml hygiene
## - Compat entry verification
##
## Reference: https://juliatesting.github.io/Aqua.jl/stable/

using Test
using Aqua
using CovarianceMatrices

@testset "Aqua.jl quality assurance" begin
    # Run Aqua tests with some checks temporarily disabled
    # These can be enabled incrementally as the codebase is cleaned up
    Aqua.test_all(
        CovarianceMatrices;
        # Skip checks that need broader refactoring
        deps_compat = false,        # TODO: Add missing compat entries for all deps
        stale_deps = false,         # TODO: Clean up unused dependencies
        unbound_args = true,
        undefined_exports = false,  # TODO: Fix undefined exports
        ambiguities = true,
        piracies = true,
        persistent_tasks = true
    )
end
