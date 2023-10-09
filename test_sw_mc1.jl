using CovarianceMatrices
using LinearAlgebra
using Random
using StatsFuns
## Monte Carlo for the regression model with a single
## stochastic regressor, in which ut and xt are independent Gaussian
## AR(1)’s with AR coefficients ρu = ρx = √0.7.

function simulate(T, beta_0, beta_1)
    # Generate the data
    rho_u = sqrt(0.7)
    rho_x = sqrt(0.7)
    sigma_u = 1
    sigma_x = 1
    u = zeros(2*T)
    x = zeros(2*T)
    for t = 2:2*T
        u[t] = rho_u * u[t-1] + sigma_u * randn()
        x[t] = rho_x * x[t-1] + sigma_x * randn()
    end
    y = beta_0 .+ beta_1 .* x + u
    return y[T+1:end], [ones(T,1) x[T+1:end]]
end



T = 600
ν = floor(Int64, 0.4*T^(2/3))
reject = zeros(10000)

for j in 1:10000

  y, x = simulate(T, 0, 0)
  β̂ = x\y
  û = y - x*β̂
  z = x.*û
  Ω̂ = CovarianceMatrices.avar(CovarianceMatrices.EWC(ν), z)
  Z = x'x
  Ω̂ᶜʰᵒˡ = cholesky(Ω̂).L
  V̂ᶜʰᵒˡ = sqrt(T).*(Z\Ω̂ᶜʰᵒˡ)
  V̂ = V̂ᶜʰᵒˡ*V̂ᶜʰᵒˡ'
  se = sqrt.(diag(V̂))

  ## sqrt.(diag(T*inv(Z)*Ω̂*inv(Z)))


  ## V̂ = cholesky(A)\cholesky(Ω̂).L
  ## V̂*V̂'
  ## inv(inv(A)*Ω̂*inv(A))/100^2
  
  t = β̂[2] ./ se[2]

  reject[j] = abs(t) >= StatsFuns.tdistinvcdf(ν, .975)
end

mean(reject)
