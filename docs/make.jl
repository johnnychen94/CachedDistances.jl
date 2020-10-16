using LazyDistances
using Documenter
using DemoCards

examples, examples_cb = makedemos("examples")

format = Documenter.HTML(;
    prettyurls=get(ENV, "CI", "false") == "true",
    canonical="https://johnnychen94.github.io/LazyDistances.jl",
    assets=String[],
)

makedocs(;
    modules=[LazyDistances],
    authors="Johnny Chen <johnnychen94@hotmail.com>",
    sitename = "LazyDistances",
    format=format,
    pages=[
        "Home" => "index.md",
        examples,
        "Function References" => "references.md"
    ],
)

examples_cb()

deploydocs(;
    repo="github.com/johnnychen94/LazyDistances.jl",
)
