##=================================================
## RegressionModel - HR
##=================================================

function aVar(k::K, m::RegressionModel) where K<:HR    
    mm = if K isa HR0 || K isa HR1
        GLM.momentmatrix(m)
    else
        X = modelmatrix(m)
        u = adjustedresiduals(k, m)
        X.*u
    end
    Σ = aVar(k, mm; unscaled=true)    
end

adjustedresiduals(k::HR0, m::RegressionModel) = residuals(m)
adjustedresiduals(k::HR1, m::RegressionModel) = residuals(m).*(1/dof_residual(m))
adjustedresiduals(k::HR2, m::RegressionModel) = residuals(m).*(1 ./ (1 .- GLM.hatvalues(m)))
adjustedresiduals(k::HR3, m::RegressionModel) = residuals(m).*(1 ./ (1 .- GLM.hatvalues(m)).^2)

function adjustedresiduals(k::HR4, m::RegressionModel)
    n, p = nobs(m), sum(.!(coef(m).==0))
    h = GLM.hatvalues(m)
    @inbounds for j in eachindex(h)
        delta = min(4.0, n*h[j]/p)
        h[j] = 1/(1-h[j])^delta
    end
    return residuals(m).*h  ## This can be doon above
end

function adjustedresiduals(k::HR4m, m::RegressionModel)
    n, p = length(response(y)), sum(.!(coef(m).==0))
    h = GLM.hatvalues(m)
    @inbounds for j in eachindex(h)
        delta = min(1, n*h[j]/p) + min(1.5, n*h[j]/p)
        h[j] = 1/(1-h[j])^delta
    end
    return residuals(m).*h
end

function adjustedresiduals(k::HR5, m::RegressionModel)
    n, p = length(response(y)), sum(.!(coef(m).==0))
    h = GLM.hatvalues(m)
    mx = max(n*0.7*maximum(h)/p, 4.0)
    @inbounds for j in eachindex(h)
        alpha = min(n*h[j]/p, mx)
        h[j] = 1/sqrt((1-h[j])^alpha)
    end
    return residuals(m).*h
end


##=================================================

## RegressionModel - CR

##=================================================
adjustment!(k::CR0, m) = 1/(length(level(k.f))-1)
adjustment!(k::CR1, m) = ((nobs(n)-1)/dof_residuls(m) * length(level(k.f))/(length(level(k.f))-1))

function adjustment(v::CR2, m::RegressionModel)    
    X, u = modelmatrix(m), residual(m)
    XX⁻¹ = crossmodelmatrix(m)
    indices = clustersindices(v)
    for j in eachindex(indices)
        Xg = view(X, index[j], :)
        ug = view(u, index[j], :)
        xAx = Symmetric(I - Xg*(XX⁻¹)*Xg')
        ldiv!(cholesky!(xAx; check=false).L, ug)
    end
    return u
end

Base.@propagate_inbounds function adjustment(k::CR3,  m::RegressionModel)
    
    X, u = modelmatrix(m), resid(c)
    n, p = size(X)
    invxx, indices = invcrossx(c), clustersindices(c)
    Threads.@threads for index in indices
        Xv = view(X, index, :)
        uv = view(u, index, :)
        xAx = Xv*invxx*Xv'
        ldiv!(cholesky!(Symmetric(I - xAx); check=false), uv)
    end
    return rmul!(u, 1/sqrt(dofadjustment(k, c)))
end



"""
    dofadjustment(k::CRHC, ::CRHCCache)

Calculate the default degrees-of-freedom adjsutment for `CRHC`

# Arguments
- `k::CRHC`: cluster robust variance type
- `c::CRHCCache`: the `CRHCCache` from which to extract the information
# Return
- `Float`: the degrees-of-fredom adjustment
# Note: the adjustment is a multyplicative factor.
"""
function dofadjustment(k::CR0, m::RegressionModel)
    g = length(clustersindices(k))::Int64
    return g/(g-1)
end

function dofadjustment(k::CR1, m::RegressionModel)
    g, (n, p) = length(clustersindices(k)), size(modelmatrix(m))
    return ((n-1)/(n-p) * g/(g-1))
end

dofadjustment(k::CR2, m::RegressionModel) = 1

function dofadjustment(k::CR3, m::RegressionModel)
     g, (n, p) = length(clustersindices(k)), size(modelmatrix(m))
    return (g/(g-1))
end
