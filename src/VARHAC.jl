# VARHAC(; maxlag=12, lagstrategy=2, selectionstrategy=:aic) = VARHAC(maxlag, lagstrategy, selectionstrategy)

# function VARHAC(maxlag::Int, lagstrategy::Int, selectionstrategy::Symbol)
#     @assert maxlag > 0 "maxlag must be greater than 0"
#     @assert lagstrategy in 1:3 "lagstrategy must be 1, 2, or 3"
#     @assert selectionstrategy in (:aic, :bic, :fixed) "selectionstrategy must be :aic, :bic, or :fixed"
#     return VARHAC(maxlag, lagstrategy, selectionstrategy)
# end

# function avar(k::VARHAC, X::Union{Matrix{F},Vector{F}}; kwargs...) where {F<:AbstractFloat}
#     #avarhac(k, X; maxlag=k.maxlag, lagstrategy=k.lagstrategy, selectionstrategy=k.selectionstrategy)
#     if k.lagstrategy == 3
#         D = fitvar(X, maxlag)
#     end
# end

# function fitvar(X, p)
#     XX = delag(X, p)
#     YY = copy(X)
#     demean_from_p!(XX, p)
#     demean_from_p!(YY, p)
#     Xv = @view XX[(p+1):end, :]
#     Yy = @view YY[(p+1):end, :]
#     A .= Xv \ Yv
#     r = Yv .- Xv * A
#     return A, r
# end

# function bic(residuals, npar, T, maxlag)
#     log.(sum(abs2, residuals, dims=1)) .+ log(T - maxlag) * npar / (T - maxlag)
# end

# function aic(residuals, npar, T, maxlag)
#     log.(sum(abs2, residuals, dims=1)) .+ 2 * npar / (T - maxlag)
# end

# function delag(X, nlags)
#     delagged = map(j -> map(x -> ShiftedArrays.lag(x, j; default=NaN), eachcol(X)), 1:nlags)
#     return reduce(hcat, collect(Base.Iterators.flatten(delagged)))
# end

# function delag_maxlag(X, maxlag)
#     T, m = size(X)
#     delagged = map(j -> map(x -> ShiftedArrays.lag(x, j; default=NaN), eachcol(view(X, maxlag:T,:))), 0:maxlag-1)
#     return reduce(hcat, collect(Base.Iterators.flatten(delagged)))
# end

# Base.@propagate_inbounds function delag!(dest, X, p::Int64)
#     n, m = size(X, 1), size(X, 2)
#     @inbounds for j ∈ Base.axes(X, 2)
#         for ℓ ∈ 1:p
#             for t ∈ (1+ℓ):n
#                 dest[t, m*(ℓ-1)+j] = X[t-ℓ, j]
#             end
#         end
#     end
# end

# Base.@propagate_inbounds function demean!(dest::VecOrMat, Y::VecOrMat)
#     μ = mean(Y; dims=1)
#     for j ∈ Base.axes(Y, 2)
#         for t ∈ Base.axes(Y, 1)
#             dest[t, j] = Y[t, j] - μ[j]
#         end
#     end
# end

# Base.@propagate_inbounds function demean_from_p!(Y::Matrix, p)
#     Yv = @view Y[(p+1):end, :]
#     μ = mean(Yv; dims=1)
#     for j ∈ Base.axes(Y, 2)
#         for t ∈ Base.axes(Yv, 1)
#             Yv[t, j] -= μ[j]
#         end
#     end
# end

# function varhac(X, lagmax)
#     ## ilag = 1
#     ## imodel = 3
#     T, m = size(X)
#     aic  = map(x->log.(sum(abs2, x .- mean(x), dims=1)), eachcol(X))
#     minres = [Vector{Float64}(undef, T-maxlag-1) for j ∈ 1:m]
#     minorder = zeros(Int64, m)
#     minpar = zeros(m, m * maxlag)
    
#     XX = delag_maxlag(X, maxlag)
#     for iorder ∈ 1:maxlag
#         Y = X[maxlag+iorder:T, :]
#         demean!(Y, Y)
#         x = XX[1+iorder:end, (iorder-1)*m+1:iorder*m]
#         demean!(x, x)
#         betas = [x\z for z in eachcol(Y)]
#         resid = [y - xv * b for b ∈ betas]
#         npar = m * iorder
#         #if imodel == 1
#             aicnew = map(x->log.(sum(abs2, x, dims=1)) .+ 2.0 * npar / (T - maxlag), resid)
#         #else
#         #    aicnew = map(x->log.(sum(abs2, x, dims=1)) .+ log(T - maxlag) * npar / (T - maxlag), resid)
#         #end
#         ind = aicnew .< aic
#         if any(ind)
#             aic[ind] .= aicnew[ind]
#             minres[ind] .= resid[ind]
#             minpar[ind, 1:m*iorder] .= betas[ind]
#             minorder[ind] .= iorder
#         end
#     end
# end
    



#     ## COMPUTE THE VARHAC ESTIMATOR
#     covar = minres'minres / (nt - imax)

#     bbb = diagm(0 => 1:kdim)

#     if imax > 0
#         for iorder = 1:imax
#             bbb = bbb - minpar[:, (iorder-1)*kdim+1:iorder*kdim]
#         end
#     end

#     inv(bbb) * covar * inv(bbb')

# end






# function varhac(dat, imax, ilag, imodel)

#     ## VARHAC ESTIMATOR FOR THE COVARIANCE MATRIX (SPECTRAL DENSITY AT FREQUENCY ZERO)

#     ## INPUT:

#     ## dat:        input matrix where each row corresponds to a different observation
#     ## imax:       maximum lag order considered
#     ##             0 = no correction for serial correlation will be made
#     ## ilag:       1 = all elements enter with the same lag
#     ##             2 = own element can enter with a different lag
#     ##             3 = only the own lag enters
#     ## imodel:     1 = AIC is used
#     ##             2 = BIC is used
#     ##             3 = a fixed lag order equal to imax is used

#     ## OUTPUT:

#     ## ccc:        VARHAC estimator

#     nt = size(dat, 1)
#     kdim = size(dat, 2)

#     ex = dat[imax:nt-1, :]
#     dat2 = dat[imax:nt-1, :]
#     ex1 = dat[imax:nt-1, :]
#     ex2 = dat[imax:nt-1, :]
#     dep = dat[imax+1:nt, 1]

#     ddd = dat[imax+1:nt, :]
#     minres = ddd
#     aic = log.(sum(minres .^ 2, dims=1))
#     minorder = zeros(kdim, 2)

#     if imax > 0
#         minpar = zeros(kdim, kdim * imax)
#         ## ALL ELEMENTS ENTER WITH THE SAME LAG (ILAG = 1)
#         if ilag == 1
#             for k = 1:kdim
#                 dep = ddd[:, k]
#                 ex = dat[imax:nt-1, :]

#                 for iorder = 1:imax
#                     if iorder == 1
#                         ex = dat[imax:nt-1, :]
#                     else
#                         ex = [ex dat[imax+1-iorder:nt-iorder, :]]
#                     end

#                     if imodel != 3
#                         ## Run the VAR
#                         b = (dep \ ex)'
#                         resid = dep - ex[:, :] * b
#                         ## Compute the model selection criterion
#                         npar = kdim * iorder
#                         if imodel == 1
#                             aicnew = log.(sum(resid .^ 2, dims=1)) .+ 2 * npar / (nt - imax)
#                         else
#                             aicnew = log.(sum(resid .^ 2, dims=1)) .+ log(nt - imax) * npar / (nt - imax)
#                         end
#                         if aicnew[1] < aic[k]
#                             aic[k] = aicnew[1]
#                             minpar[k, 1:kdim*iorder] = b'
#                             minres[:, k] = resid
#                             minorder[k] = iorder
#                         end
#                     end

#                     if imodel == 3
#                         b = (dep \ ex)'
#                         resid = dep - ex * b
#                         minpar = zeros(kdim, 2 * iorder)
#                         minpar[k, :] = b'
#                         minres[:, k] = resid
#                         minorder[k] = imax
#                     end
#                 end
#             end
#             ## ONLY THE OWN LAG ENTERS (ILAG = 3
#         elseif ilag == 3

#             for k = 1:kdim
#                 dep = ddd[:, k]

#                 for iorder = 1:imax
#                     if iorder == 1
#                         ex = dat[imax:nt-1, k]
#                     else
#                         ex = [ex dat[imax+1-iorder:nt-iorder, k]]
#                     end

#                     if imodel != 3
#                         ## Run the VAR
#                         b = ex \ dep
#                         resid = dep - ex[:, :] * b
#                         ## Compute the model selection criterion
#                         npar = iorder
#                         if imodel == 1
#                             aicnew = log.(sum(resid .^ 2, dims=1)) .+ 2 * npar / (nt - imax)
#                         else
#                             aicnew = log.(sum(resid .^ 2, dims=1)) .+ log(nt - imax) * npar / (nt - imax)
#                         end

#                         if aicnew[1] < aic[k]
#                             aic[k] = aicnew[1]
#                             for i = 1:iorder
#                                 minpar[k, k+(i-1)*kdim] = b[i]
#                             end
#                             minres[:, k] = resid
#                             minorder[k] = iorder
#                         end
#                     end
#                 end

#                 if imodel == 3
#                     b = (dep \ ex)'
#                     resid = dep - ex[:, :] * b
#                     for i = 1:imax
#                         minpar[k, k+(i-1)*kdim] = b[i]
#                     end
#                     minres[:, k] = resid
#                     minorder[k] = imax
#                 end
#             end
#             ## OWN ELEMENT CAN ENTER WITH A DIFFERENT LAG (ILAG = 2)
#         else
#             begin
#                 for k = 1:kdim
#                     dep = ddd[:, k]
#                     if k == 1
#                         dat2 = dat[:, 2:kdim]
#                     elseif k == kdim
#                         dat2 = dat[:, 1:kdim-1]
#                     else
#                         dat2 = [dat[:, 1:k-1] dat[:, k+1:kdim]]
#                     end

#                     for iorder = 0:imax
#                         if iorder == 1
#                             ex1 = dat[imax:nt-1, k]
#                         elseif iorder > 1
#                             ex1 = [ex1 dat[imax+1-iorder:nt-iorder, k]]
#                         end

#                         for iorder2 = 0:imax
#                             if iorder2 == 1
#                                 ex2 = dat2[imax:nt-1, :]
#                             elseif iorder2 > 1
#                                 ex2 = [ex2 dat2[imax+1-iorder2:nt-iorder2, :]]
#                             end

#                             if iorder + iorder2 == 0
#                                 break
#                             elseif iorder == 0
#                                 ex = ex2
#                             elseif iorder2 == 0
#                                 ex = ex1
#                             else
#                                 ex = [ex1 ex2]
#                             end
#                             ## Run the VAR
#                             b = ex \ dep
#                             #println(size(b))
#                             #println(size(ex))
#                             resid = dep .- ex[:, :] * b
#                             ## Compute the model selection criterion

#                             npar = iorder + iorder2 * (kdim - 1)

#                             if imodel == 1
#                                 aicnew = log.(sum(resid .^ 2, dims=1)) .+ 2 * npar / (nt - imax)
#                             else
#                                 aicnew = log.(sum(resid .^ 2, dims=1)) .+ log(nt - imax) * npar / (nt - imax)
#                             end
#                             #println(aic[k])
#                             if aicnew[1] < aic[k]
#                                 aic[k] = aicnew[1]
#                                 minpar[k, :] = zeros(1, kdim * imax)
#                                 for i = 1:iorder
#                                     minpar[k, k+(i-1)*kdim] = b[i]
#                                 end

#                                 for i = 1:iorder2
#                                     if k == 1
#                                         minpar[k, 2+(i-1)*kdim:kdim+(i-1)*kdim] =
#                                             b[iorder+1+(i-1)*(kdim-1):iorder+i*(kdim-1)]'
#                                     elseif k == kdim
#                                         minpar[k, 1+(i-1)*kdim:kdim-1+(i-1)*kdim] =
#                                             b[iorder+1+(i-1)*(kdim-1):iorder+i*(kdim-1)]'
#                                     else
#                                         minpar[k, 1+(i-1)*kdim:k-1+(i-1)*kdim] =
#                                             b[iorder+1+(i-1)*(kdim-1):iorder+k-1+(i-1)*(kdim-1)]'
#                                         minpar[k, k+1+(i-1)*kdim:kdim+(i-1)*kdim] =
#                                             b[iorder+k+(i-1)*(kdim-1):iorder+i*(kdim-1)]'
#                                     end
#                                 end
#                                 minres[:, k] = resid
#                                 minorder[k, :] = [iorder iorder2]
#                             end
#                         end
#                     end
#                 end
#             end
#         end
#     end

#     ## COMPUTE THE VARHAC ESTIMATOR
#     covar = minres'minres / (nt - imax)

#     bbb = diagm(0 => 1:kdim)

#     if imax > 0
#         for iorder = 1:imax
#             bbb = bbb - minpar[:, (iorder-1)*kdim+1:iorder*kdim]
#         end
#     end

#     inv(bbb) * covar * inv(bbb')

# end
