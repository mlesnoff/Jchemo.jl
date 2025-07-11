"""
    samprand(n::Int, k::Int; seed::Union{Nothing, Int} = nothing)
    samprand(group::Vector, k::Int; seed::Union{Nothing, Int} = nothing)
Build training vs. test sets by random sampling.  
* `n` : Total nb. of observations.
* `group` : A vector (`n`) defining groups of observations.
* `k` : Nb. test observations, or nb. test groups if `group` is used, returned in each 
    validation segment.
Keyword arguments:
* `seed` : Eventual seed for the `Random.MersenneTwister` generator. 

Two outputs are returned (= row indexes of the data): 
* `train` (`n` - `k`),
* `test` (`k`). 

If `group` is used (must be a vector of length `n`), the function samples groups of observations instead 
of single observations. Such a group-sampling is required when the data are structured by groups 
and when the response to predict is correlated within groups. This prevents underestimation of the 
generalization error.

## Examples
```julia
using Jchemo

n = 10
samprand(n, 4)
samprand(n, 4; seed = 123)

n = 10 
group = ["A", "B", "C", "D", "E", "A", "B", "C", "D", "E"]    # groups of the observations
tab(group)  
k = 2 
res = samprand(group, k)
group[res.test]
```
""" 
function samprand(n::Int, k::Int; seed::Union{Nothing, Int} = nothing)
    zn = collect(1:n)
    if isnothing(seed)
        s = StatsBase.sample(zn, k; replace = false)
    else
        s = StatsBase.sample(MersenneTwister(seed), zn, k; replace = false) 
    end 
    sort!(s)
    train = zn[setdiff(1:end, s)]
    (train = train, test = s)
end

function samprand(group::Vector, k::Int; seed::Union{Nothing, Int} = nothing)
    s = segmts(group, k; rep = 1, seed)[1][1]
    zn = collect(1:length(group))
    train = rmrow(zn, s)  
    test = zn[s]
    (train = train, test)
end

