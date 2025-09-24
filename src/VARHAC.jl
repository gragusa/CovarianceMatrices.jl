
function avar(
        k::VARHAC{S, L},
        X::AbstractMatrix{R};
        kwargs...
) where {S <: LagSelector, L <: SameLags, R <: Real}
    lagstrategy = isa(k.selector, AICSelector) ? :aic : :bic
    Ω, AICs,
    BICs,
    order_aic,
    order_bic = _var_selection_samelag(X, maxlags(k)...; lagstrategy = lagstrategy, demean = false)
    k.AICs = AICs
    k.BICs = BICs
    k.order_aic = order_aic
    k.order_bic = order_bic
    return Ω
end

function avar(
        k::VARHAC{S, L},
        X::AbstractMatrix{R};
        kwargs...
) where {S <: LagSelector, L <: DifferentOwnLags, R <: Real}
    lagstrategy = isa(k.selector, AICSelector) ? :aic : :bic
    Ω, AICs,
    BICs,
    order_aic,
    order_bic = _var_selection_ownlag(X, maxlags(k)...; lagstrategy = lagstrategy, demean = false)
    k.AICs = AICs
    k.BICs = BICs
    k.order_aic = order_aic
    k.order_bic = order_bic
    return Ω
end

function avar(
        k::VARHAC{S, L},
        X::AbstractMatrix{R};
        kwargs...
) where {S <: LagSelector, L <: FixedLags, R <: Real}
    lagstrategy = isa(k.selector, AICSelector) ? :aic : :bic
    Ω, AICs, BICs, order_aic, order_bic = _var_fixed(X, maxlags(k)...; demean = false)
    k.AICs = AICs
    k.BICs = BICs
    k.order_aic = order_aic
    k.order_bic = order_bic
    return Ω
end

function _var_selection_samelag(
        X::AbstractMatrix{R},
        K;
        lagstrategy::Symbol = :aic,
        demean::Bool = false
) where {R <: Real}
    T, m = size(X)
    ## ---------------------------------------------------------
    ## Demean the data if requested
    ## ---------------------------------------------------------
    if demean == true
        Y = X .- mean(X; dims = 1)
    else
        Y = copy(X)
    end
    ## ---------------------------------------------------------
    ## Matrix of lags
    ## ---------------------------------------------------------
    Z = delag(Y, K)
    ## ---------------------------------------------------------
    ## Containers
    ## ---------------------------------------------------------
    order_aic = zeros(Int, m)
    order_bic = zeros(Int, m)
    𝕏β = Vector{R}(undef, T - K)
    ε = Vector{R}(undef, T - K)

    ## ---------------------------------------------------------
    ## Calculate the AIC & BIC for each variable at lag 0
    ## ---------------------------------------------------------
    𝕐 = view(Y, (K + 1):T, :)
    RSS = sum(abs2, 𝕐; dims = 1)
    AIC = vec(log.(RSS / T))
    BIC = vec(log.(RSS / T))
    AICs = Array{R}(undef, m, K)
    BICs = similar(AICs)
    ## ---------------------------------------------------------
    ## Calculate AIC & BIC for each variable at lags 1,2,...,K
    ## ---------------------------------------------------------
    @inbounds for k in 1:K
        𝕏 = view(Z, :, 1:(k * m))
        𝕏𝕏 = Matrix{R}(undef, k * m, k * m)
        𝕏𝕐 = Vector{R}(undef, k * m)
        for j in axes(Y, 2)
            𝕐 = view(Y, (K + 1):T, j)
            mul!(𝕏𝕏, 𝕏', 𝕏)
            mul!(𝕏𝕐, 𝕏', 𝕐)
            ## -----------------------
            ## Perform OLS coefficient
            ## -----------------------
            β = cholesky!(Symmetric(𝕏𝕏)) \ 𝕏𝕐
            ## -----------------------
            ## Calculate the residuals
            ## -----------------------
            mul!(𝕏β, 𝕏, β)
            ε .= 𝕐 .- 𝕏β
            ## -----------------------
            ## Calculate the AIC&BIC
            ## -----------------------
            RSS = sum(abs2, ε)
            AIC_ = 2 * k * m / T + log(RSS / T)
            BIC_ = log(T) * k * m / T + log(RSS / T)
            ## -----------------------
            ## Update the AIC and BIC
            ## -----------------------
            AICs[j, k] = AIC_
            BICs[j, k] = BIC_
            AIC_ < AIC[j] && (AIC[j] = AIC_; order_aic[j] = k)
            BIC_ < BIC[j] && (BIC[j] = BIC_; order_bic[j] = k)
        end
    end
    ## Estimate VAR with the order selected by AIC and BIC
    if lagstrategy == :aic
        order = order_aic
    elseif lagstrategy == :bic
        order = order_bic
    end

    A = zeros(R, m, m, K)
    ε = Array{R}(undef, T, m)
    @inbounds for h in 1:m
        ε[1:order[h], h] .= NaN
    end
    @inbounds for j in axes(Y, 2)
        if order[j] > 0
            𝕐 = view(Y, (order[j] + 1):T, j)
            𝕏 = delag(X, order[j])
            β = cholesky!(Symmetric(𝕏'𝕏)) \ 𝕏'𝕐
            𝕏β = 𝕏 * β
            ε[(order[j] + 1):end, j] .= 𝕐 .- 𝕏β
            A[j, :, 1:order[j]] = β
        else
            copy!(view(ε, :, j), Y[:, j])
        end
    end
    Γ = pinv(I - dropdims(sum(A; dims = 3); dims = 3))
    B = nancov(ε; corrected = false)
    return Γ * B * Γ, AICs, BICs, order_aic, order_bic
end

function _var_selection_ownlag(
        X::AbstractMatrix{R},
        K,
        Kₓ;
        lagstrategy::Symbol = :aic,
        demean::Bool = false
) where {R <: Real}
    ## K is the maximum own lag
    ## Kₓ is the maximum cross lag
    T, m = size(X)
    ## ---------------------------------------------------------
    ## Demean the data if requested
    ## ---------------------------------------------------------
    if demean == true
        Y = X .- mean(X; dims = 1)
    else
        Y = copy(X)
    end
    ## ---------------------------------------------------------
    ## Matrix of lags
    ## ---------------------------------------------------------
    maxK = max(K, Kₓ)
    Z = delag(Y, maxK)
    ## ---------------------------------------------------------
    ## Containers
    ## ---------------------------------------------------------
    order_aic = zeros(Int, m, 2)
    order_bic = zeros(Int, m, 2)
    AICs = Array{R}(undef, m, K+1, Kₓ+1)
    BICs = similar(AICs)
    𝕏β = Vector{R}(undef, T - maxK)
    ε = Vector{R}(undef, T - maxK)

    ## ---------------------------------------------------------
    ## Calculate the AIC & BIC for each variable at lag 0
    ## ---------------------------------------------------------
    𝕐 = view(Y, (K + 1):T, :)
    RSS = sum(abs2, 𝕐; dims = 1)
    AIC = vec(log.(RSS / T))
    BIC = copy(AIC)
    AICs[:, 1, 1] .= AIC
    BICs[:, 1, 1] .= BIC
    @show "LAGS..: 0"
    @show "AIC...: ", AIC
    ## ---------------------------------------------------------
    ## Calculate AIC & BIC for each variable at lags 1,2,...,K
    ## ---------------------------------------------------------
    @inbounds for kₓ in 0:Kₓ
        for k in 0:K
            #- `m::Int`: The number of columns in the original matrix X.
            #- `K::Int`: The maximum number of lags used to create matrix Z.
            #- `position_own::Int`: The index of the column (1 ≤ position_own ≤ m) for which a different number of lags will be selected.
            #- `lags_others::Int`: The number of lags to select for all columns except the 'position_own' column (1 ≤ lags_others ≤ K).
            #- `lags_own::Int`: The number of lags to select for the 'position_own' column (1 ≤ lags_own ≤ K).
            𝕏𝕏 = Matrix{R}(undef, kₓ * (m - 1) + k, kₓ * (m - 1) + k)
            𝕏𝕐 = Vector{R}(undef, kₓ * (m - 1) + k)
            for j in axes(Y, 2)
                𝕐 = view(Y, (maxK + 1):T, j)
                𝕏 = select_lags(Z, m, maxK, j, k, kₓ)
                mul!(𝕏𝕏, 𝕏', 𝕏)
                mul!(𝕏𝕐, 𝕏', 𝕐)
                ## -----------------------
                ## Perform OLS coefficient
                ## -----------------------
                β = cholesky!(Symmetric(𝕏𝕏)) \ 𝕏𝕐
                ## -----------------------
                ## Calculate the residuals
                ## -----------------------
                mul!(𝕏β, 𝕏, β)
                ε .= 𝕐 .- 𝕏β
                ## -----------------------
                ## Calculate the AIC&BIC
                ## -----------------------
                RSS = sum(abs2, ε)
                AIC_ = 2 * (kₓ * (m - 1) + k) / T + log(RSS / T)
                BIC_ = log(T) * (kₓ * (m - 1) + k) / T + log(RSS / T)
                @show "LAGS..: ", k, kₓ
                @show "AIC...: ", AIC_
                ## -----------------------
                ## Update the AIC and BIC
                ## -----------------------
                AICs[j, k + 1, kₓ + 1] = AIC_
                BICs[j, k + 1, kₓ + 1] = BIC_
                AIC_ < AIC[j] && (AIC[j] = AIC_; order_aic[j, :] .= [k, kₓ])
                BIC_ < BIC[j] && (BIC[j] = BIC_; order_bic[j, :] .= [k, kₓ])
            end
        end
    end
    ## Estimate VAR with the order selected by AIC and BIC
    if lagstrategy == :aic
        order = order_aic
    elseif lagstrategy == :bic
        order = order_bic
    end
    ε = Array{R}(undef, T, m)
    @inbounds for h in 1:m
        ε[1:sum(order[h, :]), h] .= NaN
    end
    maxK = maximum(sum(order, dims = 2))
    A = zeros(R, m, m*maxK)

    @inbounds for j in axes(Y, 2)
        kk = sum(order[j, :])
        ℤ = delag(X, kk)
        if kk > 0
            𝕐 = view(Y, (kk + 1):T, j)
            𝕏 = select_lags(ℤ, m, kk, j, order[j, :]...)
            β = cholesky!(Symmetric(𝕏'𝕏)) \ 𝕏'𝕐
            𝕏β = 𝕏 * β
            ε[(kk + 1):end, j] .= 𝕐 .- 𝕏β
            A[j, 𝕏.indices[2]] = β
        else
            copy!(view(ε, :, j), Y[:, j])
        end
    end
    𝔸 = reshape(A, (m, m, maxK))
    Γ = pinv(I - dropdims(sum(𝔸; dims = 3); dims = 3))
    B = nancov(ε; corrected = false)
    return Γ * B * Γ, AICs, BICs, order_aic, order_bic
end

function _var_fixed(X::AbstractMatrix{R}, K; demean::Bool = false) where {R <: Real}
    ## K is the maximum own lag
    ## Kₓ is the maximum cross lag
    T, m = size(X)
    ## ---------------------------------------------------------
    ## Demean the data if requested
    ## ---------------------------------------------------------
    if demean == true
        Y = X .- mean(X; dims = 1)
    else
        Y = copy(X)
    end
    ## ---------------------------------------------------------
    ## Matrix of lags
    ## ---------------------------------------------------------
    Z = delag(Y, K)
    ## ---------------------------------------------------------
    ## Containers
    ## ---------------------------------------------------------
    𝕐 = view(Y, (K + 1):T, :)
    A = Z\𝕐
    ε = 𝕐 .- Z * A
    𝔸 = reshape(A', (m, m, K))
    Γ = pinv(I - dropdims(sum(𝔸; dims = 3); dims = 3))
    B = nancov(ε; corrected = false)
    return Γ * B * Γ, [], [], [K], [K]
end

function delag(X::Matrix{R}, K::Int) where {R <: Real}
    T, n = size(X)
    Z = Matrix{Float64}(undef, T-K, n*K)
    @inbounds for j in 1:n
        for t in (K + 1):T
            for k in 1:K
                Z[t - K, (k - 1) * n + j] = X[t - k, j]
            end
        end
    end
    return Z
end

"""
    select_lags(Z::Matrix{T}, m::Int, K::Int, position_own::Int, lags_own::Int, lags_others::Int) where T<:Real

Create an efficient view of matrix Z, selecting specific lags of certain columns based on the input parameters.

# Arguments
- `Z::Matrix{T}`: The input matrix of size (T-K) × (mK), where T is the number of rows in the original matrix X,
                  K is the maximum number of lags, and m is the number of columns in X.
- `m::Int`: The number of columns in the original matrix X.
- `K::Int`: The maximum number of lags used to create matrix Z.
- `position_own::Int`: The index of the column (1 ≤ position_own ≤ m) for which a different number of lags will be selected.
- `lags_others::Int`: The number of lags to select for all columns except the 'position_own' column (1 ≤ lags_others ≤ K).
- `lags_own::Int`: The number of lags to select for the 'position_own' column (1 ≤ lags_own ≤ K).

# Returns
- `view`: A view of Z containing the selected lags of the specified columns.

# Details
This function creates a view of Z that includes:
1. `lags_others` lags of columns 1,2,...,position_own-1,position_own+1,...,m from the original matrix X.
2. `lags_own` lags of column `position_own` from the original matrix X.

# Notes
- The resulting view will have (T-K) rows and (lags_others*(m-1) + lags_own) columns.

# Example
```julia
Z = rand(100, 30)
# Assuming Z is created from a 5-column matrix X with 6 lags
m, K = 5, 6
position_own, lags_others, lags_own = 3, 4, 5
result = select_lags(Z, m, K, position_own, lags_others, lags_own)
```
"""
function select_lags(
        Z::Matrix{T},
        m::Int,
        K::Int,
        position_own::Int,
        lags_own::Int,
        lags_others::Int
) where {T <: Real}
    # Check if the dimensions are correct
    s = position_own
    r = lags_others
    v = lags_own
    (T_K, mK) = size(Z)
    @assert mK == m * K "The number of columns in Z should be m * K"
    @assert r <= K && v <= K "r and v should not exceed K"
    @assert s <= m "s should not exceed m"
    # Calculate the indices for the r lags of columns 1,2,...,s,s+2,...,m
    r_indices = vcat(
        [((k-1)*m .+ (1:(s - 1))) for k in 1:r]...,
        [((k-1)*m .+ ((s + 1):m)) for k in 1:r]...
    )
    # Calculate the indices for the v lags of column s
    v_indices = [(k-1)*m .+ s for k in 1:v]

    # Combine all indices
    all_indices = sort!(union(r_indices, v_indices))

    # Return the view
    return view(Z, :, all_indices)
end
