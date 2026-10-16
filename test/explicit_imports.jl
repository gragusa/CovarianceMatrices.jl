## ExplicitImports.jl checks
##
## Guards against implicit imports, stale explicit imports, import/access
## ownership regressions, and reliance on non-public names.
##
## Allowed non-public names:
## - `Clustering`, `clusterize`, `fit_var`, `_checkclusterlength`: the SparseArrays
##   extension imports this internal type, adds methods to these internal functions
##   of CovarianceMatrices itself, and reuses its cluster-length check.
## - `_hatweights`: the GLM extension adds methods to this internal function.
## - `Base.print_matrix`: `show(::IO, ::MIME"text/plain", ::CovarianceMatrix)`
##   uses it to print the matrix body without the `Matrix{T}` header line;
##   Base has no public equivalent.
## - `Base.@deprecate_binding`: retires an exported name while keeping it working;
##   Base has no public equivalent.
## - `LinearAlgebra.LAPACK.syev!`: CR2 eigendecomposes many small symmetric matrices, and
##   for those `syev!` is faster than the `syevr!` behind `eigen(::Symmetric)`.
##   LinearAlgebra documents its LAPACK wrappers but does not mark them public.
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
            ignore = (:print_matrix, Symbol("@deprecate_binding"), :clusterize, :fit_var,
            :_checkclusterlength, :_hatweights, :syev!)
        )
    )
end
