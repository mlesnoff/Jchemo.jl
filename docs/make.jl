#push!(LOAD_PATH,joinpath(@__DIR__, ".."))
#push!(LOAD_PATH,"../src/")
using Documenter
using Jchemo
#using UMAP

DocMeta.setdocmeta!(Jchemo, :DocTestSetup, :(using Jchemo); recursive = true)

makedocs(;
    modules = [Jchemo],
    authors = "Matthieu Lesnoff",
    repo = Documenter.Remotes.GitHub("mlesnoff", "Jchemo.jl"),
    #repo = "https://github.com/mlesnoff/Jchemo.jl/blob/{commit}{path}#L{line}",
    sitename = "Jchemo.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://mlesnoff.github.io/Jchemo.jl",
        edit_link = "main",
        assets = String[],
        example_size_threshold = 500000,
        size_threshold = 2_500_000,
        size_threshold_warn = 2_500_000,
        ),
    pages = [
        "Home" => "index.md",
        "Available methods" => "domains.md",
        "Index of functions" => "api.md",
        "News" => "news.md"
        #"Examples" => "see_jchemodemo.md",
        #"Datasets" => "see_jchemodata.md"
        ]
    )

deploydocs(;
    repo = "github.com/mlesnoff/Jchemo.jl.git" #,
    #devbranch = "main"
    )
