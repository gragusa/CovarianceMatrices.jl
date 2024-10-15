invpseudohessian(x) = (t = typeof(x); error("Not defined for type $t", ))

"""
### Description
The `momentmatrix` function returns the matrix of moment conditions for the estimation problem defined by `x`. Moment conditions are crucial for various estimation procedures, such as Generalized Method of Moments (GMM) and Maximum Likelihood Estimation (MLE). 

For MLE, the moment matrix corresponds to the inverse of the Hessian of the (pseudo-)log-likelihood function, evaluated at the data. For GMM, it represents the matrix of moment conditions evaluated at the observed data.

This function can be extended to support user-defined types, allowing flexibility for different estimation methods.

### Usage

```julia
momentmatrix(x) -> Matrix
momentmatrix(x, t) -> Matrix
momentmatrix!(x, y) -> Matrix
```

- `momentmatrix(x)`: Returns the moment matrix for the estimation problem `x`.
- `momentmatrix(x, t)`: Returns the moment matrix for the estimation problem `x` when the parameter is equal to t.
- `momentmatrix!(x, y)`: In-place version, updating the matrix `y` with the moment conditions evaluated for `x`.

The matrix returned is typically of size `(obs x m)`, where `obs` refers to the number of observations, and `m` refers to the number of moments. Users can define their own moment matrices for custom types by overloading this function.
"""
momentmatrix(x) = (t = typeof(x); error("Not defined for type $t"))
momentmatrix!(x, y) = (t = typeof(x); error("Not defined"))
resid(x) = (t = typeof(x); error("Not defined for type $t"))
"""
    bread(x)
Return the bread matrix for the estimation problem `x`.

Note: This function is not defined for all types and must be extended for specific types.
"""
bread(x) = (t = typeof(x); error("Not defined for type $t"))
leverage(x) = (t = typeof(x); error("Not defined for type $t"))
residualadjustment(k::AVarEstimator, x::Any) = (t = typeof(x); error("Not defined for type $t"))

function StatsBase.vcov(𝒦::AVarEstimator, e)
    gᵢ= momentmatrix(e)
    ## Bread mut return a k×m
    B = bread(e)
    Ω = aVar(𝒦, gᵢ)
    B*Ω*B'/size(gᵢ,1)
end

StatsBase.stderror(𝒦::AVarEstimator, e) = sqrt.(diag(vcov(𝒦, e)))
