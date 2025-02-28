"""
    segmts(n::Int, m::Int; rep = 1, seed = nothing)
    segmts(group::Vector, m::Int; rep = 1, seed = nothing)
Build segments of observations for "test-set" validation.
* `n` : Total nb. of observations in the dataset. 
    The sampling  is implemented within 1:`n`.
* `group` : A vector (n) defining blocks of observations.
* `m` : Nb. test observations, or groups if `group` is used, returned 
    in each segment.
Keyword arguments: 
* `rep` : Nb. replications of the sampling.
* `seed` : Eventual seed for the `Random.MersenneTwister` 
    generator. Must be of length = `rep`. When `nothing`, 
    the seed is random at each replication.
    
For each replication, the function builds a test set that can 
be used to validate a model. 

If `group` is used (must be a vector of length n), the function 
samples entire groups (= blocks) of observations instead of observations. 
Such a block-sampling is required when data is structured by blocks and 
when the response to predict is correlated within blocks. 
This prevents underestimation of the generalization error.

The function returns a list (vector) of `rep` elements. 
Each element of the list is a vector of the indexes (positions 
within 1:`n`) of the sampled observations.  

## Examples
```julia
using Jchemo 

n = 10 ; m = 3
rep = 4 
segm = segmts(n, m; rep) 
i = 1
segm[i]
segm[i][1]

n = 10 
group = ["A", "B", "C", "D", "E", "A", "B", "C", "D", "E"]    # blocks of the observations
tab(group)  
m = 2 ; rep = 4 
segm = segmts(group, m; rep)
i = 1 
segm[i]
segm[i][1]
group[segm[i][1]]
```
""" 
function segmts(n::Int, m::Int; rep = 1, seed = nothing)
    Q = Vector{Int}
    s = list(Vector{Q}, rep)
    for i = 1:rep
        s[i] = list(Q, 1)
        if isnothing(seed)
            s[i][1] = StatsBase.sample(1:n, m; replace = false, ordered = true)
        else
            s[i][1] = StatsBase.sample(MersenneTwister(Int(seed[i])), 1:n, m; replace = false, ordered = true)
        end
    end
    s
end

function segmts(group::Vector, m::Int; rep = 1, seed = nothing)
    group = vec(group)
    Q = Vector{Int}
    s = list(Vector{Q}, rep)
    yagg = unique(group)
    nlev = length(yagg)
    m = min(m, nlev)
    @inbounds for i = 1:rep
        s[i] = list(Q, 1)
        if isnothing(seed)
            s[i][1] = StatsBase.sample(1:nlev, m; replace = false, ordered = true)
        else
            s[i][1] = StatsBase.sample(MersenneTwister(Int(seed[i])), 1:nlev, m; replace = false, ordered = true)
        end
    end
    zs = copy(s)
    @inbounds for i = 1:rep
        u = s[i][1]
        v = findall(in(yagg[u]).(group))        # which(group %in% yagg[u])
        zs[i][1] = v
    end
    zs    
end

