#push!(LOAD_PATH, joinpath(@__DIR__, ".."))
#push!(LOAD_PATH,"../src/")

using Documenter
using Jchemo

DocMeta.setdocmeta!(Jchemo, :DocTestSetup, :(using Jchemo); recursive = true)

makedocs(

# Build documentation
makedocs(;
    modules = [Jchemo],
    doctest = true,
    clean = false,
    sitename = "Jchemo.jl",
    authors = "Matthieu Lesnoff (Cirad, Montpellier, France)",
    format = Documenter.HTML(
        #prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://mlesnoff.github.io/Jchemo.jl/stable/",
        assets = String[],
        edit_link = "main"
        ),
        pages = ["Domains" => "functions.md",
        "Index" => "api.md",
        "News" => "news.md"],
    strict = true
    )

# Deploy built documentation from Travis
deploydocs(;
    repo = "github.com/mlesnoff/Jchemo.jl.git",
    target = "build",
    deps = nothing,
    make = nothing,
    devbranch = "main"
    )

