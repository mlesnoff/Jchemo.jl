"""
    segmts(n::Int, k::Int; rep = 1, seed::Union{Nothing, Int, Vector{Int}} = nothing)
    segmts(group::Vector, k::Int; rep = 1, seed::Union{Nothing, Int, Vector{Int}} = nothing)
Build segments of observations for "test-set" validation.
* `n` : Total nb. of observations in the dataset. The sampling is implemented within 1:`n`.
* `group` : A vector (`n`) defining groups of observations.
* `k` : Nb. test observations, or nb. test groups if `group` is used, returned in each 
    validation segment.
Keyword arguments: 
* `rep` : Nb. replications of the sampling.
* `seed` : Eventual seed for the `Random.MersenneTwister` generator. Must be of length = `rep`. 
    
For each replication, the function builds a test set that can be used to validate a model. 

If `group` is used (must be a vector of length `n`), the function samples groups of observations instead 
of single observations. Such a group-sampling is required when the data are structured by groups 
and when the response to predict is correlated within groups. This prevents underestimation of the 
generalization error.

The function returns a list (vector) of `rep` elements. Each element of the list is a vector of the 
indexes (positions within 1:`n`) of the sampled observations.  

## Examples
```julia
using Jchemo 

n = 10 ; k = 3
rep = 4 
segm = segmts(n, k; rep) 
i = 1
segm[i]
segm[i][1]

segmts(n, k; seed = 123)
segmts(n, k; rep, seed = collect(1:rep))

n = 10 
group = ["A", "B", "C", "D", "E", "A", "B", "C", "D", "E"]    # groups of the observations
tab(group)  
k = 2 ; rep = 4 
segm = segmts(group, k; rep)
i = 1 
segm[i]
segm[i][1]
group[segm[i][1]]
```
""" 
function segmts(n::Int, k::Int; rep = 1, seed::Union{Nothing, Int, Vector{Int}} = nothing)
    Q = Vector{Int}
    s = list(Vector{Q}, rep)
    for i = 1:rep
        s[i] = list(Q, 1)
        if isnothing(seed)
            s[i][1] = StatsBase.sample(1:n, k; replace = false, ordered = true)
        else
            s[i][1] = StatsBase.sample(MersenneTwister(seed[i]), 1:n, k; replace = false, ordered = true)
        end
    end
    s
end

function segmts(group::Vector, k::Int; rep = 1, seed::Union{Nothing, Int, Vector{Int}} = nothing)
    group = vec(group)
    Q = Vector{Int}
    s = list(Vector{Q}, rep)
    yagg = unique(group)
    nlev = length(yagg)
    k = min(k, nlev)
    @inbounds for i = 1:rep
        s[i] = list(Q, 1)
        if isnothing(seed)
            s[i][1] = StatsBase.sample(1:nlev, k; replace = false, ordered = true)
        else
            s[i][1] = StatsBase.sample(MersenneTwister(seed[i]), 1:nlev, k; replace = false, ordered = true)
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

