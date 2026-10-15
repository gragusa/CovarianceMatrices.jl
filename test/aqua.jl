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

@testset "Aqua.jl" begin
    Aqua.test_all(CovarianceMatrices)
end
