using LiveServer

# LiveServer expects to run from the package root so it can find docs/make.jl.
cd(joinpath(@__DIR__, "..")) do
    servedocs(; launch_browser = get(ENV, "CI", nothing) != "true")
end
