##=================================================

## GLM

##=================================================

adjustment(k::HR2, m::RegressionModel) = 1 ./(1 .- hatmatrix(m))
adjustment(k::HR3, m::RegressionModel) = 1 ./(1 .- hatmatrix(m)).^2

function adjustment(k::HR4, m::RegressionModel)
    n, p = nobs(m), sum(.!(coef(m).==0))
    h = hatmatrix(m)
    @inbounds for j in eachindex(h)
        delta = min(4.0, n*h[j]/p)
        h[j] = 1/(1-h[j])^delta
    end
    return h
end

function adjustment(k::HR4m, m::RegressionModel)
    n, p = length(response(y)), sum(.!(coef(m).==0))
    h = hatmatrix(m, x)
    @inbounds for j in eachindex(h)
        delta = min(1, n*h[j]/p) + min(1.5, n*h[j]/p)
        h[j] = 1/(1-h[j])^delta
    end
    return h
end

function adjustment(k::HR5, m::RegressionModel)
    n, p = length(response(y)), sum(.!(coef(m).==0))
    h = hatmatrix(m)
    mx = max(n*0.7*maximum(h)/p, 4.0)
    @inbounds for j in eachindex(h)
        alpha = min(n*h[j]/p, mx)
        h[j] = 1/sqrt((1-h[j])^alpha)
    end
    return h
end
