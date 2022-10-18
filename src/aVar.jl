# """


#  aVar(k::AVarEstimators, M::Matrix{T}; demean=true, prewhite=false, dof=0.0)

# Caclulate the asymptotic variance of \bar{X}_{,j}=\frac{1}{b_n}\sum{i=1}^n X_{i,j}, where \bar{X}_{,j} is the j-th column of `X`.
#  The assumption is b_n is 

#  ## Arguments 
#     k::CovarianceEstimator 
#         - `dims=1`: the dimension along which the variables are organized. When dims = 1, 
#     the variables are considered columns with observations in rows; when dims = 2, variables are in rows with 
#     observations in columns. 
#         - `demean=false`: whether the data should be demeaned. 
    
    
#     - `mean=nothing` known mean value. `nothing` indicates that the mean is unknown, and the function will compute the mean if `demean=true`. 
#        If `demean=false` indicates that the data are centered and hence there's no need to subtract the mean (even if it is provided). 
    
#     - `prewhite=false`: should the data be prewithened. Relevant for `HAC` estimator.
    
#     - `unscaled=false`: When false the scale is `n` for HR and HAC and `G` for CR. 
#       When true do not attempt to scale the variance estimator so it simply return sum
# """
    
function aVar(k::AVarEstimator, X::AbstractMatrix{T}; demean::Bool=true, prewhiten::Bool=false, dof::Int64=0, mean::Any=nothing, unscaled::Bool = false) where T<:AbstractFloat
    Z = if demean 
            if mean===nothing 
                mean = Statistics.mean(X; dims = 1)
            end 
            X .- mean
        else 
            X 
    end 
    
    if k isa HAC && prewhiten
        Z, D = prewhiter(Z)
        k.prewhiten.x = true        
    end
    Shat = avar(k, Z)
    k isa HAC && prewhiten ? dewhiter!(Shat, Z, D) : nothing
    return !unscaled ? Shat./size(X,1) : Shat
end