## ExplicitImports.jl checks
##
## Guards against implicit imports, stale explicit imports, import/access
## ownership regressions, and reliance on non-public names.
##
## Allowed non-public names:
## - `Clustering`, `clusterize`, `fit_var`, `_checkclusterlength`: the SparseArrays
##   extension imports this internal type, adds methods to these internal functions
##   of CovarianceMatrices itself, and reuses its cluster-length check.
## - `Base.print_matrix`: `show(::IO, ::MIME"text/plain", ::CovarianceMatrix)`
##   uses it to print the matrix body without the `Matrix{T}` header line;
##   Base has no public equivalent.
##
## Reference: https://ericphanson.github.io/ExplicitImports.jl/stable/

using Test
using ExplicitImports
using CovarianceMatrices

@testset "ExplicitImports" begin
    test_explicit_imports(
        CovarianceMatrices;
        all_explicit_imports_are_public = (; ignore = (:Clustering,)),
        all_qualified_accesses_are_public = (;
            ignore = (:print_matrix, :clusterize, :fit_var, :_checkclusterlength)
        )
    )
end
