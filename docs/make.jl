#push!(LOAD_PATH, joinpath(@__DIR__, ".."))
#push!(LOAD_PATH,"../src/")

using Documenter
using Jchemo

DocMeta.setdocmeta!(Jchemo, :DocTestSetup, :(using Jchemo); recursive = true)

makedocs(;
    modules = [Jchemo],
    doctest = true,
    clean = false,
    sitename = "Jchemo.jl",
    authors = "Matthieu Lesnoff (Cirad, Montpellier, France)",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://mlesnoff.github.io/Jchemo.jl",
        assets = String[]
        ),
    pages = ["Domains" => "functions.md",
        "Index" => "api.md",
        "News" => "news.md"]
    )

deploydocs(;
    repo = "github.com/mlesnoff/Jchemo.jl.git",
    devbranch = "main"
    )

