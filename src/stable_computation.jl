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
function _compute_vcov(
        form::Information,
        H,
        G,
        Ω,
        W;
        cond_atol::Union{Nothing, Real} = nothing,
        cond_rtol::Union{Nothing, Real} = nothing,
        debug::Bool = false,
        warn::Bool = true
)
    # Set default tolerances if not provided
    atol = cond_atol === nothing ? 0.0 : cond_atol
    rtol_val = cond_rtol === nothing ? (eps(real(float(eltype(Ω)))) * min(size(Ω)...)) :
               cond_rtol

    # Force warn=true when debug=true
    warn = debug || warn

    # Check if we have a Hessian (MLE case) or need to use score (GMM case)
    if H !== nothing
        # MLE case: V = H⁻¹ (Fisher Information)
        Hinv, flag, svals = ipinv(H; atol = atol, rtol = rtol_val)
        _debug_report_inversion("H (Hessian)", flag, svals, "$(size(H))", debug, warn)
        return Hinv
    else
        # GMM case: V = (G'Ω⁻¹G)⁻¹ (efficient GMM)
        Ωinv, flag_omega, svals_omega = ipinv(Ω; atol = atol, rtol = rtol_val)
        _debug_report_inversion("Omega", flag_omega, svals_omega, "$(size(Ω))", debug, warn)

        G_omega_G = G' * Ωinv * G
        V, flag_v, svals_v = ipinv(G_omega_G; atol = atol, rtol = rtol_val)
        _debug_report_inversion("G'Omega^(-1)G (variance matrix)", flag_v,
            svals_v, "$(size(G_omega_G))", debug, warn)
        return V
    end
end

# Misspecified form - dispatches based on model type context
# For MLE-like models (m=k): robust sandwich V = G⁻¹ΩG⁻ᵀ
# For GMM-like models (m>k): robust GMM V = (G'WG)⁻¹(G'WΩWG)(G'WG)⁻¹

function _compute_vcov(
        form::Misspecified,
        H,
        G,
        Ω,
        W;
        cond_atol::Union{Nothing, Real} = nothing,
        cond_rtol::Union{Nothing, Real} = nothing,
        debug::Bool = false,
        warn::Bool = true
)
    m, k = size(G)

    # Set default tolerances if not provided
    atol = cond_atol === nothing ? 0.0 : cond_atol
    rtol_val = cond_rtol === nothing ? (eps(real(float(eltype(Ω)))) * min(size(Ω)...)) :
               cond_rtol

    # Force warn=true when debug=true
    warn = debug || warn

    if m == k
        # MLE-like: robust sandwich form V = G⁻¹ΩG⁻ᵀ
        Ginv, flag, svals = ipinv(G; atol = atol, rtol = rtol_val)
        _debug_report_inversion("G (score matrix)", flag, svals, "$(size(G))", debug, warn)
        return Ginv' * Ω * Ginv
    else
        # GMM-like: robust GMM form V = (G'WG)⁻¹(G'WΩWG)(G'WG)⁻¹
        _compute_vcov_gmm_misspecified(H, G, Ω, W; cond_atol = cond_atol,
            cond_rtol = cond_rtol, debug = debug, warn = warn)
    end
end

function _debug_report_inversion(matrix_name::String, flag::AbstractVector{Bool},
        svals::Vector, size_info::String, debug::Bool, warn::Bool)
    if any(flag) && !isempty(svals)
        n_problematic = sum(flag)
        min_sval = minimum(svals)
        max_sval = maximum(svals)
        problematic_svals = svals[findall(flag)]  # Use findall for proper indexing

        if debug
            println("🔍 DEBUG: Matrix inversion issue detected")
            println("   Matrix: $matrix_name ($size_info)")
            println("   Problematic singular values: $n_problematic/$(length(svals))")
            println("   Singular value range: [$(min_sval), $(max_sval)]")
            if !isempty(problematic_svals)
                println("   Problematic values: $(sort(problematic_svals))")
            end
            println("   Condition number: $(max_sval / max(min_sval, eps()))")
            println()
        elseif warn
            @warn "The inverse of $matrix_name is not invertible. Correction to the smallest eigenvalues applied ($(n_problematic)/$(length(svals)) values affected)"
        end
    end
end

function _compute_vcov_gmm_misspecified(
        H,
        G,
        Ω,
        W;
        cond_atol::Union{Nothing, Real} = nothing,
        cond_rtol::Union{Nothing, Real} = nothing,
        debug::Bool = false,
        warn::Bool = true
)
    # Set default tolerances if not provided
    atol = cond_atol === nothing ? 0.0 : cond_atol
    rtol_val = cond_rtol === nothing ? (eps(real(float(eltype(Ω)))) * min(size(Ω)...)) :
               cond_rtol

    # Force warn=true when debug=true
    warn = debug || warn

    ## Invert matrices with debug reporting
    Ωinv, flag_omega, svals_omega = ipinv(Ω; atol = atol, rtol = rtol_val)
    _debug_report_inversion("Ω", flag_omega, svals_omega, "$(size(Ω))", debug, warn)

    G_omega_G = G' * Ωinv * G
    B, flag_gomega, svals_gomega = ipinv(G_omega_G; atol = atol, rtol = rtol_val)
    _debug_report_inversion(
        "G'Omega^(-1)G", flag_gomega, svals_gomega, "$(size(G_omega_G))", debug, warn)

    Hinv, flag_h, svals_h = ipinv(H; atol = atol, rtol = rtol_val)
    _debug_report_inversion("H", flag_h, svals_h, "$(size(H))", debug, warn)

    return Hinv' * B * Hinv
end

function ipinv(
        A::AbstractMatrix{T};
        atol::Real = 0.0,
        rtol::Real = (eps(real(float(oneunit(T)))) * min(size(A)...)) * iszero(atol)
) where {T}
    m, n = size(A)
    Tout = typeof(zero(T) / sqrt(oneunit(T) + oneunit(T)))
    if m == 0 || n == 0
        return similar(A, Tout, (n, m)), Bool[], T[]
    end
    if isdiag(A)
        indA = LinearAlgebra.diagind(A)
        dA = view(A, indA)
        maxabsA = maximum(abs, dA)
        tol = max(rtol * maxabsA, atol)
        B = fill!(similar(A, Tout, (n, m)), 0)
        indB = LinearAlgebra.diagind(B)
        B[indB] .= (x -> abs(x) > tol ? pinv(x) : zero(x)).(dA)
        problematic = .!(abs.(dA) .> tol)
        return B, problematic, abs.(dA)  # Return diagonal values as "singular values"
    end
    SVD = svd(A)
    tol2 = max(rtol * maximum(SVD.S), atol)
    Stype = eltype(SVD.S)
    Sinv = fill!(similar(A, Stype, length(SVD.S)), 0)
    index = SVD.S .> tol2
    Sinv[index] .= pinv.(view(SVD.S, index))
    problematic = .!index
    return SVD.Vt' * (Diagonal(Sinv) * SVD.U'), problematic, SVD.S
end

function ipinv(x::Number)
    xi = inv(x)
    return ifelse(isfinite(xi), xi, zero(xi))
end
