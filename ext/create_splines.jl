using CovarianceMatrices
using Dierckx
using JLD2

basepath = @__DIR__

x = 0.1:.01:32
itr = Iterators.product(x, 1:175000)
out = map(i -> CovarianceMatrices.kernel(QuadraticSpectralKernel(1), i[2]/i[1]), itr)
r7 = vec(convert(Array{Float64}, mapslices(u -> findfirst(abs.(u) .< 1e-07), out, dims = 2)))
r8 = vec(convert(Array{Float64}, mapslices(u -> findfirst(abs.(u) .< 1e-08), out, dims = 2)))
r9 = vec(convert(Array{Float64}, mapslices(u -> findfirst(abs.(u) .< 1e-09), out, dims = 2)))

spline7 = Spline1D(collect(x), r7)
spline8 = Spline1D(collect(x), r8)
spline9 = Spline1D(collect(x), r9)

x0 = collect(x)
spval7 = Dierckx.evaluate(spline7, x0)
spval8 = Dierckx.evaluate(spline8, x0)
spval9 = Dierckx.evaluate(spline9, x0)

@show maximum(abs(spval7-r7))
@show maximum(abs(spval8-r8))
@show maximum(abs(spval9-r9))

@save "qs_spline_e7.jld2" spline7
@save "qs_spline_e8.jld2" spline8
@save "qs_spline_e9.jld2" spline9
