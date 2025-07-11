"""
    samprand(n::Int, k::Int; replace = false, seed::Union{Nothing, Int} = nothing)
Build training vs. test sets by random sampling.  
* `n` : Total nb. of observations.
* `k` : Nb. test observations to sample.
Keyword arguments:
* `replace` : Boolean. If `false`, the sampling is without replacement.
* `seed` : Eventual seed for the `Random.MersenneTwister` generator. 

Two outputs are returned (= row indexes of the data): 
* `train` (`n` - `k`),
* `test` (`k`). 

Output `test` is built by random sampling within `1:n`. 

## Examples
```julia
using Jchemo

n = 10
samprand(n, 4)
```
""" 
function samprand(n::Int, k::Int; replace = false, seed::Union{Nothing, Int} = nothing)
    zn = collect(1:n)
    if isnothing(seed)
        s = StatsBase.sample(zn, k; replace = replace)
    else
        s = StatsBase.sample(MersenneTwister(seed), zn, k; replace = replace) 
    end 
    sort!(s)
    train = zn[setdiff(1:end, s)]
    (train = train, test = s)
end

