#!/bin/bash
#=
export __JULIA_SPAWNED__=1
exec julia --color=yes --startup-file=no "${BASH_SOURCE[0]}"
=#

if get(ENV, "__JULIA_SPAWNED__", "0") != "1"
    @info "Spawning new Julia process"
    let file = @__FILE__
        try
            ENV["__JULIA_SPAWNED__"] = "1"
            run(`$(Base.julia_cmd()) $file`)
        finally
            ENV["__JULIA_SPAWNED__"] = "0"
        end
    end
else
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()

    using Coverage
    cd(joinpath(@__DIR__, "..", "..")) do
        coverage = process_folder()
        infofile = joinpath(@__DIR__, "coverage-lcov.info")
        LCOV.writefile(infofile, coverage)

        outdir = joinpath(@__DIR__, "html")
        rm(outdir, recursive=true, force=true)
        mkdir(outdir)
        cmd = Sys.iswindows() ? "genhtml.cmd" : "genhtml"
        run(`$cmd $infofile --output-directory=$outdir`)
    end
end

nothing

# Local Variables:
# mode: julia
# End:
