using CachedDistances
using Documenter

format = Documenter.HTML(;
    prettyurls=get(ENV, "CI", "false") == "true",
    canonical="https://johnnychen94.github.io/CachedDistances.jl",
    assets=String[],
)

makedocs(;
    modules=[CachedDistances],
    authors="Johnny Chen <johnnychen94@hotmail.com>",
    sitename = "CachedDistances",
    format=format,
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/johnnychen94/CachedDistances.jl",
)
