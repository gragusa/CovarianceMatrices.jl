using Documenter, CovarianceMatrices

makedocs(
    sitename = "CovarianceMatrices.jl",
    authors = "Giuseppe Ragusa and contributors",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://gragusa.github.io/CovarianceMatrices.jl/stable/",
        assets = String[],
        edit_link = "master",
        sidebar_sitename = false
    ),
    modules = [CovarianceMatrices],
    checkdocs = :exports,
    clean = true,
    pages = [
        "Home" => "index.md",
        "Introduction & Mathematical Foundation" => "introduction.md",
        "Estimators" => [
            "HAC Estimators" => "estimators/hac.md",
            "Heteroskedasticity-Robust (HC/HR)" => "estimators/hc.md",
            "Clustered Standard Errors (CR)" => "estimators/cr.md",
            "VARHAC" => "estimators/varhac.md",
            "Smoothed Moments" => "estimators/smoothed_moments.md",
            "Driscoll-Kraay" => "estimators/driscoll_kraay.md",
            "EWC" => "estimators/ewc.md"
        ],
        "Tutorials" => [
            "Matrix Interface" => "tutorials/matrix_tutorial.md",
            "GLM Integration" => "tutorials/glm_tutorial.md",
            "Package Interface Extension" => "tutorials/interface_tutorial.md"
        ],
        "API Reference" => "api.md",
        "Performance Notes" => "performance.md"
    ]
)

deploydocs(
    repo = "github.com/gragusa/CovarianceMatrices.jl.git",
    target = "build",
    branch = "gh-pages",
    versions = ["stable" => "v^", "v#.#"]
)
