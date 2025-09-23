"""
Numerically stable variance computation methods.

This module implements the core variance computations using factorizations
instead of explicit matrix inversions for numerical stability.
"""

using LinearAlgebra

"""
    _compute_vcov(form::VarianceForm, model, Ω, W; kwargs...)

Dispatch to appropriate variance computation based on form.
"""
function _compute_vcov(form::Information, model, Ω, W; rcond_tol::Real=1e-12, warn::Bool=true)
    return _compute_information(model; rcond_tol=rcond_tol, warn=warn)
end

function _compute_vcov(form::Robust, model, Ω, W; rcond_tol::Real=1e-12, warn::Bool=true)
    G = jacobian(model)
    return _compute_robust(G, Ω; rcond_tol=rcond_tol)
end

function _compute_vcov(form::CorrectlySpecified, model, Ω, W; rcond_tol::Real=1e-12, warn::Bool=true)
    G = jacobian(model)
    return _compute_correctly_specified(G, Ω; rcond_tol=rcond_tol)
end

function _compute_vcov(form::Misspecified, model, Ω, W; rcond_tol::Real=1e-12, warn::Bool=true)
    G = jacobian(model)

    # Determine weight matrix
    if W === nothing
        W_eff = _compute_optimal_weight(Ω; rcond_tol=rcond_tol)
        if warn
            @warn "Misspecified form with W=nothing reduces to CorrectlySpecified"
        end
    else
        W_eff = W
    end

    return _compute_misspecified(G, Ω, W_eff; rcond_tol=rcond_tol)
end

"""
    _compute_information(model; rcond_tol::Real=1e-12, warn::Bool=true)

Compute V = H⁻¹ for information matrix form.
"""
function _compute_information(model; rcond_tol::Real=1e-12, warn::Bool=true)
    # Try to get objective Hessian first
    H = objective_hessian(model)

    if H === nothing
        # Fall back to Jacobian if it's symmetric (and square)
        G = jacobian(model)
        if G === nothing
            throw(ArgumentError("Information form requires either objective_hessian or jacobian"))
        end

        m, k = size(G)
        if m != k
            throw(ArgumentError("Information form requires exactly identified model (m = k)"))
        end

        if !issymmetric(G)
            if warn
                @warn "Using non-symmetric Jacobian as Hessian for Information form"
            end
        end
        H = G
    end

    return _safe_inverse(H; rcond_tol=rcond_tol)
end

"""
    _compute_robust(G::AbstractMatrix, Ω::AbstractMatrix; rcond_tol::Real=1e-12)

Compute V = G⁻¹ΩG⁻ᵀ for robust sandwich form.
"""
function _compute_robust(G::AbstractMatrix, Ω::AbstractMatrix; rcond_tol::Real=1e-12)
    m, k = size(G)
    if m != k
        throw(ArgumentError("Robust form requires exactly identified model (m = k)"))
    end

    # Use QR decomposition for numerical stability
    F = qr(G)

    # Solve G⁻¹Ω
    X = F \ Ω  # X = G⁻¹Ω

    # Compute V = X * G⁻ᵀ = G⁻¹ΩG⁻ᵀ
    V = X / F.R'  # Equivalent to X * (G⁻¹)ᵀ

    return V
end

"""
    _compute_correctly_specified(G::AbstractMatrix, Ω::AbstractMatrix; rcond_tol::Real=1e-12)

Compute V = (G'Ω⁻¹G)⁻¹ without forming Ω⁻¹ explicitly.
"""
function _compute_correctly_specified(G::AbstractMatrix, Ω::AbstractMatrix; rcond_tol::Real=1e-12)
    # Factor Ω
    F_Ω = _safe_factorize(Ω; rcond_tol=rcond_tol)

    # Solve ΩX = G → X = Ω⁻¹G
    X = F_Ω \ G

    # Compute K = G'Ω⁻¹G = G'X
    K = Symmetric(G' * X)

    # Return K⁻¹
    return _safe_inverse(K; rcond_tol=rcond_tol)
end

"""
    _compute_misspecified(G::AbstractMatrix, Ω::AbstractMatrix, W::AbstractMatrix; rcond_tol::Real=1e-12)

Compute V = (G'WG)⁻¹(G'WΩWG)(G'WG)⁻¹ for misspecified GMM.
"""
function _compute_misspecified(G::AbstractMatrix, Ω::AbstractMatrix, W::AbstractMatrix; rcond_tol::Real=1e-12)
    # Compute K = G'WG
    WG = W * G
    K = Symmetric(G' * WG)

    # Compute B = G'WΩWG
    WΩW = W * Ω * W
    B = Symmetric(G' * WΩW * G)

    # Factor K
    F_K = _safe_factorize(K; rcond_tol=rcond_tol)

    # Solve K⁻¹B
    X = F_K \ B

    # Compute V = XK⁻¹ = K⁻¹BK⁻¹
    if F_K isa Cholesky
        V = X / F_K.U  # For Cholesky factorization
    else
        V = F_K \ X'  # For other factorizations, then transpose
        V = V'
    end

    return V
end

"""
    _compute_optimal_weight(Ω::AbstractMatrix; rcond_tol::Real=1e-12)

Compute optimal weight matrix W = Ω⁻¹.
"""
function _compute_optimal_weight(Ω::AbstractMatrix; rcond_tol::Real=1e-12)
    F = _safe_factorize(Ω; rcond_tol=rcond_tol)
    return inv(F)
end

"""
    _safe_factorize(A::AbstractMatrix; rcond_tol::Real=1e-12)

Safely factorize a symmetric positive definite matrix with SVD fallback.
"""
function _safe_factorize(A::AbstractMatrix; rcond_tol::Real=1e-12)
    try
        # Try Cholesky first (fastest for SPD)
        return cholesky(Symmetric(A))
    catch PosDefException
        # Fall back to LDL for symmetric matrices
        try
            return ldlt(Symmetric(A))
        catch
            # Final fallback to SVD-based pseudoinverse
            @warn "Matrix is not positive definite; using SVD pseudoinverse"
            return _svd_pseudoinverse(A; rcond_tol=rcond_tol)
        end
    end
end

"""
    _safe_inverse(A::AbstractMatrix; rcond_tol::Real=1e-12)

Safely compute matrix inverse with SVD fallback for ill-conditioned matrices.
"""
function _safe_inverse(A::AbstractMatrix; rcond_tol::Real=1e-12)
    try
        # Try direct factorization
        F = _safe_factorize(A; rcond_tol=rcond_tol)
        if F isa SVDPseudoInverse
            return F.pinv
        else
            return inv(F)
        end
    catch
        # SVD fallback
        return _svd_pseudoinverse(A; rcond_tol=rcond_tol).pinv
    end
end

# Helper struct for SVD-based pseudoinverse
struct SVDPseudoInverse{T}
    pinv::Matrix{T}
    rank::Int
end

function _svd_pseudoinverse(A::AbstractMatrix{T}; rcond_tol::Real=1e-12) where T
    F = svd(A)
    σ_cutoff = rcond_tol * F.S[1]
    keep = F.S .> σ_cutoff
    rank_used = sum(keep)

    if rank_used < length(F.S)
        @warn "Matrix is rank deficient (rank $(rank_used)/$(length(F.S))); using pseudoinverse"
    end

    # Compute pseudoinverse
    σ_inv = zeros(T, length(F.S))
    σ_inv[keep] .= 1 ./ F.S[keep]

    pinv = F.V * Diagonal(σ_inv) * F.U'

    return SVDPseudoInverse(pinv, rank_used)
end

# Make SVDPseudoInverse work with LinearAlgebra operations
Base.:\(F::SVDPseudoInverse, b) = F.pinv * b
LinearAlgebra.inv(F::SVDPseudoInverse) = F.pinv

"""
    _scale_vcov!(V::AbstractMatrix, scale::Symbol, n::Int)

Apply scaling to variance-covariance matrix.
"""
function _scale_vcov!(V::AbstractMatrix, scale::Symbol, n::Int)
    if scale == :n
        V ./= n
    elseif scale == :none
        # No scaling
    else
        throw(ArgumentError("Unknown scale option: $scale. Use :n or :none"))
    end
    return V
end