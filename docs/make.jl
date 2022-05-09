push!(LOAD_PATH,joinpath(@__DIR__, ".."))
#push!(LOAD_PATH,"../src/")
using Documenter, Jchemo

### PlantGeom.jl 
#DocMeta.setdocmeta!(Jchemo, :DocTestSetup, :(using Jchemo); recursive = true)
### End

makedocs(
    modules = [Jchemo],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    ### PlantGeom.jul
    #repo = "https://github.com/mlesnoff/Jchemo.jl/blob/{commit}{path}#{line}",
    #format = Documenter.HTML(;
    #    prettyurls = get(ENV, "CI", "false") == "true",
    #    canonical = "https://mlesnoff.github.io/Jchemo.jl",
    #    assets = String[]
    #    ),
    ### End
    authors = "Matthieu Lesnoff",
    sitename = "Jchemo.jl",
    pages = ["Domains" => "functions.md",
        "Index" => "api.md",
        "News" => "news.md"]
    # strict = true,
    # clean = true,
    # checkdocs = :exports,
    )

deploydocs(
    repo = "github.com/mlesnoff/Jchemo.jl.git",
    )
