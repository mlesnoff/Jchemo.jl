"""
    sampbag(n, p; rep = 50, rowsamp = .7, replace = true, colsamp = 1)
    sampbag(n, p, colweight::ProbabilityWeights; rep = 50, rowsamp = .7, replace = true, colsamp = 1)
Sampling for bagging.
* `n`, `p` : Nb. total of observations and variables, respectively, considered in the bagging.
* `colweight` : Weights (p) of the variables. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).
Keyword arguments:
* `rep` : Number of replications of the bagging.
* `rowsamp` : Proportion of observations to sample within `n at each replication`.
* `replace`: Boolean. If `true`, observations are sampled with replacement.
* `colsamp`: Proportion of observations to sample within `p` (without replacement) at each replication.

## Examples
```julia
using Jchemo  

n = 10 ; p = 4 ; q = 2
res = sampbag(n, p; rep = 4, rowsamp = .7, colsamp = .7) ;
@names res
res.srow
res.srow_oob
res.scol
```
""" 
function sampbag(n, p; rep = 50, rowsamp = .7, replace = true, colsamp = 1)
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
    Threads.@threads for i = 1:rep 
        ## Rows
        s = StatsBase.sample(range_n, mrow; replace, ordered)
        srow[i] = s
        srow_oob[i] = range_n[setdiff(1:end, s)]
        ## Columns
        if colsamp == 1
            scol[i] = range_p
        else
            s = StatsBase.sample(range_p, mcol; replace = false, ordered)
            scol[i] = s
        end
    end
    (srow = srow, srow_oob, scol)
end

function sampbag(n, p, colweight::ProbabilityWeights; rep = 50, rowsamp = .7, replace = true, colsamp = 1)
    range_n = collect(1:n)
    range_p = collect(1:p) 
    mrow = Int(round(rowsamp * n))
    mcol = max(1, Int(round(colsamp * p)))    
    ##
    srow = list(Vector{Int}, rep)
    srow_oob = list(Vector{Int}, rep)
    scol = list(Vector{Int}, rep)
    ##
    Threads.@threads for i = 1:rep 
        ## Rows
        s = StatsBase.sample(range_n, mrow; replace, ordered = true)
        srow[i] = s
        srow_oob[i] = range_n[setdiff(1:end, s)]
        ## Columns
        if colsamp == 1
            scol[i] = range_p
        else
            colweight.values[colweight.values .== 0] .= eps(eltype(colweight.values))
            colweight = pweight(colweight.values)
            s = StatsBase.sample(range_p, colweight, mcol; replace = false, ordered = true)
            scol[i] = s
        end
    end
    (srow = srow, srow_oob, scol)
end


