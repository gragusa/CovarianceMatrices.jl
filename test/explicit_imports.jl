## ExplicitImports.jl checks
##
## Guards against implicit imports, stale explicit imports, and import/access
## ownership regressions.
##
## The two public-ness checks are disabled: the SparseArrays extension
## legitimately reaches into CovarianceMatrices' own internals (`Clustering`,
## `clusterize`, `fit_var`), which are deliberately non-public.
##
## Reference: https://ericphanson.github.io/ExplicitImports.jl/stable/

using Test
using ExplicitImports
using CovarianceMatrices

@testset "ExplicitImports" begin
    test_explicit_imports(
        CovarianceMatrices;
        all_explicit_imports_are_public = false,
        all_qualified_accesses_are_public = false,
    )
end
