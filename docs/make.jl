## NOT USED AT PRESENT

push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using Documenter, Jchemo

makedocs(
    modules = [Jchemo],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    authors = "Matthieu Lesnoff",
    sitename = "Jchemo.jl",
    pages = Any["index.md",
        "regression.md"]
    # strict = true,
    # clean = true,
    # checkdocs = :exports,
)

deploydocs(
    repo = "github.com/mlesnoff/Jchemo.jl.git",
)
