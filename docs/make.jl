#push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using Documenter
using Jchemo

makedocs(
    modules = [Jchemo],
    authors = "Matthieu Lesnoff (Cirad, Montpellier, France)",
    repo = "https://github.com/mlesnoff/Jchemo.jl/blob/{commit}{path}#{line}",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://mlesnoff.github.io/Jchemo.jl",
        assets = String[])
    #format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    sitename = "Jchemo.jl",
    pages = ["News" => "news.md",
        "Functions" => "functions.md",
        "API" => "api.md"]
    )

deploydocs(; repo = "github.com/mlesnoff/Jchemo.jl.git")
