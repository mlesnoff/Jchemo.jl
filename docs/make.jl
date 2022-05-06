#push!(LOAD_PATH, joinpath(@__DIR__, ".."))
#push!(LOAD_PATH,"../src/")
using Documenter
using Jchemo

DocMeta.setdocmeta!(Jchemo, :DocTestSetup, :(using Jchemo); recursive = true)

makedocs(
    modules = [Jchemo],
    sitename = "Jchemo.jl",
    authors = "Matthieu Lesnoff (Cirad, Montpellier, France)",
    #repo = "https://mlesnoff.github.io/Jchemo.jl/blob/{commit}{path}#{line}",
    #format = Documenter.HTML(;
    #    prettyurls = get(ENV, "CI", "false") == "true",
    #    canonical = "https://mlesnoff.github.io/Jchemo.jl",
    #    assets = String[]),
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    pages = ["Domains" => "functions.md",
        "Index" => "api.md",
        "News" => "news.md"]
    )

deploydocs( 
    repo = "github.com/mlesnoff/Jchemo.jl.git"
    )
