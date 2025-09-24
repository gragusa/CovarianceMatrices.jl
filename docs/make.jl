using Documenter, CovarianceMatrices

makedocs(
    modules = [CovarianceMatrices],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    clean = true,
    sitename = "CovarianceMatrices.jl",
    authors = "Giuseppe Ragusa",
    checkdocs = :exports,
    # strict = true,
    pages = [
        "Introduction" => "introduction.md",
        #"Long run covariances" => "lrcov.md",
        #"GLM covariances" => "glmcov.md"])a
    ],
)

deploydocs(repo = "github.com/gragusa/CovarianceMatrices.jl.git")
