"""
    sampbag(n::Int, p::Int; rep::Int = 50, replace::Bool = true, rowsamp::Q = .7, 
        colsamp::Q = .7, seed::Union{Nothing, Int} = nothing) where Q <: Float 
    sampbag(n::Int, p::Int, colweight::ProbabilityWeights{Q}; rep::Int = 50, replace::Bool = true, 
        rowsamp::Q = .7, colsamp::Q = .7, seed::Union{Nothing, Int} = nothing) where Q <: Float 
Sampling for bagging.
* `n`, `p` : Nb. total of observations and variables, respectively, considered in the bagging.
* `colweight` : Weights (p) of the variables. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).
Keyword arguments:
* `rep` : Number of replications of the bagging.
* `replace`: Boolean. If `true`, observations are sampled with replacement.
* `rowsamp` : Proportion of observations to sample within `n at each replication`.
* `colsamp`: Proportion of observations to sample within `p` (without replacement) at each replication.
* `seed` : Eventual seed for the `Random.MersenneTwister` generator used to select rows and columns.

# Examples
```julia
using Jchemo  

n = 10 ; p = 4 ; q = 2
res = sampbag(n, p; rep = 4, rowsamp = .7, colsamp = .7) ;
#res = sampbag(n, p; rep = 4, rowsamp = .7, colsamp = .7, seed = 1234) ;
@names res
res.srow
res.srow_oob
res.scol
```
""" 
function sampbag(n::Int, p::Int; rep::Int = 50, replace::Bool = true, rowsamp::Q = .7, 
        colsamp::Q = .7, seed::Union{Nothing, Int} = nothing) where Q <: Float 
    range_n = collect(1:n)
    range_p = collect(1:p) 
    mrow = Int(round(rowsamp * n))
    mcol = max(1, Int(round(colsamp * p)))    
    ##
    srow = list(Vector{Int}, rep)
    srow_oob = list(Vector{Int}, rep)
    scol = list(Vector{Int}, rep)
    ##
    ordered = true
    Threads.@threads for i in eachindex(srow)
        # Rows
        vseed = isnothing(seed) ? seed : seed + i - 1   
        s = StatsBase.sample(MersenneTwister(vseed), range_n, mrow; replace, ordered)
        srow[i] = s
        srow_oob[i] = range_n[setdiff(1:end, s)]
        # Columns
        if colsamp == 1
            scol[i] = range_p
        else
            s = StatsBase.sample(MersenneTwister(vseed), range_p, mcol; replace = false, ordered)
            scol[i] = s
        end
    end
    (srow = srow, srow_oob, scol)
end

function sampbag(n::Int, p::Int, colweight::ProbabilityWeights{Q}; rep::Int = 50, replace::Bool = true, 
        rowsamp::Q = .7, colsamp::Q = .7, seed::Union{Nothing, Int} = nothing) where Q <: Float 
    range_n = collect(1:n)
    range_p = collect(1:p) 
    mrow = Int(round(rowsamp * n))
    mcol = max(1, Int(round(colsamp * p)))    
    ##
    srow = list(Vector{Int}, rep)
    srow_oob = list(Vector{Int}, rep)
    scol = list(Vector{Int}, rep)
    ##
    ordered = true
    Threads.@threads for i in eachindex(srow)
        # Rows
        vseed = isnothing(seed) ? seed : seed + i - 1   
        s = StatsBase.sample(MersenneTwister(vseed), range_n, mrow; 
            replace, ordered)
        srow[i] = s
        srow_oob[i] = range_n[setdiff(1:end, s)]
        # Columns
        if colsamp == 1
            scol[i] = range_p
        else
            colweight.values[colweight.values .== 0] .= eps(eltype(colweight.values))
            colweight = pweight(colweight.values)
            s = StatsBase.sample(MersenneTwister(vseed), range_p, colweight, mcol; 
                replace = false, ordered)
            scol[i] = s
        end
    end
    (srow = srow, srow_oob, scol)
end


