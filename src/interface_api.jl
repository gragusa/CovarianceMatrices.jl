# ## ------------------------------------------------------
# ## Interface
# ## ------------------------------------------------------
# invpseudohessian(x) = (t = typeof(x); error("Not defined for type $t", ))
# momentmatrix(x) = (t = typeof(x); error("Not defined for type $t"))
# resid(x) = (t = typeof(x); error("Not defined for type $t"))
# ## ------------------------------------------------------
# ## HAC
# ## ------------------------------------------------------
# function setupkernelweights!(k::T, m) where T<:HAC
#     cf = coef(m)
#     resize!(k.weights, length(cf))
#     fill!(k.weights, 1)
# end

# setupkernelweights!(k::T, m::F) where {T<:Smoothed, F} = return nothing

# function StatsBase.vcov(k::T, m; prewhite=false, dof_adjustment::Bool=true, scale::Real=1) where T<:Union{HAC, Smoothed}
#     return _vcov(k, m, prewhite, dof_adjustment, scale)
# end

# function _vcov(k::T, m, prewhite::Bool, dof_adjustment::Bool, scale::Real) where T<:Union{HAC, Smoothed}
#     setupkernelweights!(k, m)
#     B  = invpseudohessian(m)
#     mm = momentmatrix(m)
#     A = lrvar(k, mm; prewhite=prewhite, demean=false)
#     scale *= (dof_adjustment ? nobs(m)^2/dof_residual(m) : nobs(m))
#     V = Symmetric((B*A*B).*scale)
#     return V
# end

# vcovmatrix(
#     k::T,
#     m,
#     factorization=Cholesky;
#     prewhite=false,
#     dof_adjustment::Bool=true,
#     scale::Real=1,
# ) where T<:Union{HAC, Smoothed} = _vcovmatrix(k, m, prewhite, dof_adjustment, scale, factorization)

# function _vcovmatrix(
#     k::T,
#     m,
#     prewhite::Bool,
#     dof_adjustment::Bool,
#     scale::Real,
#     ::Type{Cholesky},
# ) where T<:Union{HAC, Smoothed}
#     V = _vcov(k, m, prewhite, dof_adjustment, scale)
#     return CovarianceMatrix(cholesky(V, check=true), k, V)
# end

# function _vcovmatrix(
#     k::T,
#     m,
#     prewhite::Bool,
#     dof_adjustment::Bool,
#     scale::Real,
#     ::Type{SVD},
# ) where T<:Union{HAC, Smoothed}
#     V = _vcov(k, m, prewhite, dof_adjustment, scale)
#     return CovarianceMatrix(svd(V), k, V)
# end

# ## ------------------------------------------------------
# ## HC
# ## Note: adjfactor is always n/(n-k) for H1-H5 types
# ## ------------------------------------------------------
# adjfactor(k::HC0, m, x) = one(first(x))
# adjfactor(k::HC1, m, x) = nobs(m)./dof_residual(m)

# function adjfactor(k::T, m, x) where T<:Union{HC2, HC3, HC4, HC4m, HC5}
#     mm = typeof(m)
#     @warn "$k variance is not defined for type $mm. Using `HC1`."
#     numobs(m)./dof_residual(m)
# end

# adjust!(m, adj::AbstractFloat) = m
# adjust!(m, adj::AbstractMatrix) = m.*sqrt.(adj)

# StatsBase.vcov(k::HC, m::T; scale::Real=1) where T = _vcov(k, m, scale)

# vcovmatrix(k::HC, m::T, factorization = Cholesky; scale::Real = 1)  where T = _vcovmatrix(k, m, scale, factorization)

# function _vcovmatrix(k::HC, m::T, scale::Real, ::Type{Cholesky}) where T
#     V = _vcov(k, m, scale)
#     CovarianceMatrix(cholesky(V, check=true), k, V)
# end

# function _vcovmatrix(k::HC, m::T, scale::Real, ::Type{SVD}) where T
#     V = _vcov(k, m, scale)
#     CovarianceMatrix(svd(V), k, V)
# end

# ## This needs to be here - because for GLM 
# ## we used modmatrix and some other indirection
# ## to get care of weights and dispersion
# ## TODO: It could be fixed by more cleverly dealing 
# ##       with the weights situation
# function _vcov(k::HC, m::T, scale) where T
#     B   = invpseudohessian(m)
#     mm  = momentmatrix(m)
#     adj = adjfactor(k, m, modelmatrix(m))  ## <- here `modelmatrix` in GLM.jl `modmatrix`
#     mm  = adjust!(mm, adj)
#     scale *= length(adj) > 1 ? one(first(adj)) : adj
#     A = mm'*mm
#     return Symmetric((B*A*B).*scale)
# end


# ## ------------------------------------------------------
# ## CRHC
# ## ------------------------------------------------------

# StatsBase.vcov(k::CRHC, m::T; scale::Real=1) where T = _vcov(k, m, scale)

# vcovmatrix(k::CRHC, m::T, factorization = Cholesky; scale::Real = 1)  where T = _vcovmatrix(k, m, scale, factorization)

# function _vcovmatrix(k::CRHC, m::T, scale::Real, ::Type{Cholesky}) where T
#     V = _vcov(k, m, scale)
#     CovarianceMatrix(cholesky(V, check=true), k, V)
# end

# function _vcovmatrix(k::CRHC, m::T, scale::Real, ::Type{SVD}) where T
#     V = _vcov(k, m, scale)
#     CovarianceMatrix(svd(V), k, V)
# end

# ## This needs to be here - because for GLM 
# ## we used modmatrix and some other indirection
# ## to get care of weights and dispersion
# ## TODO: It could be fixed by more cleverly dealing 
# ##       with the weights situation
# function _vcov(k::CRHC, m::T, scale) where T
#     mm  = momentmatrix(m)
#     C = installcache(k, mm)
#     A = clusterize!(C)
#     B = invpseudohessian(m)
#     return Symmetric((B*A*B))
# end

# function _vcov!(C::CRHCCache, m::T) where T
#     mm  = momentmatrix(m)
#     C.momentmatrix .= mm
#     A = clusterize!(C)
#     B = invpseudohessian(m)
#     return Symmetric((B*A*B))
# end

# """
# Calculate sandwich type variance-covariance of an estimator
# """
# function sandwich(k::RobustVariance, B::Matrix, momentmatrix::Matrix; demean = false, prewhite = false, dof = 0.0)    
#     K, m = size(B)    
#     n, m_m = size(momentmatrix)
#     @assert m_m == m "number of rows of `momentmatrix` must be equal to the number of column of `B`"
#     scale = n^2/(n-dof)
#     A = lrvar(k, momentmatrix; demean = demean, scale = scale, prewhite = prewhite)   # df adjustment is built into vcov
#     Symmetric(B*A*B)
# end

# ## Standard errors
# StatsBase.stderror(k::RobustVariance, m; kwargs...) = sqrt.(diag(vcov(k, m; kwargs...)))
# StatsBase.stderror(k::Smoothed, m; kwargs...) = sqrt.(diag(vcov(k, m; kwargs...)))
# StatsBase.stderror(v::CovarianceMatrix) = sqrt.(diag(v.V))
