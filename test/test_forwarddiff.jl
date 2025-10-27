using ForwardDiff, CovarianceMatrices, LinearAlgebra

X = randn(100, 5)
y = X*[1;2;3;4;5]./100 + randn(100)

d = (y=y, X=X)

g(θ, d) = d.X.*(d.y - d.X*θ)

function obj(θ, d)
    gg = g(θ, d)
    Ω = aVar(EWC(10), gg)
    gn = sum(gg, dims=1)
    only(gn * inv(Ω) * gn')
end

obj(zeros(5), d)

ForwardDiff.gradient(θ -> obj(θ, d), zeros(5))



function obj(θ, d)
    gg = g(θ, d)
    Ω = aVar(UniformSmoother(10), gg)
    gn = sum(gg, dims=1)
    only(gn * inv(Ω) * gn')
end

obj(zeros(5), d)

ForwardDiff.gradient(θ -> obj(θ, d), zeros(5))


function obj(θ, d)
    gg = g(θ, d)
    Ω = aVar(Bartlett(10), gg)
    gn = sum(gg, dims=1)
    only(gn * inv(Ω) * gn')
end

obj(zeros(5), d)
ForwardDiff.gradient(θ -> obj(θ, d), zeros(5))