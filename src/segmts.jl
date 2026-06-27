"""
    segmts(n::Signed, k::Signed; rep = 1, seed::Union{Nothing, Int} = nothing)
    segmts(group::Vector, k::Signed; rep = 1, seed::Union{Nothing, Int} = nothing)
Build segments of observations for "test-set" validation.
* `n` : Total nb. of observations in the dataset. The sampling is implemented within 1:`n`.
* `group` : A vector (`n`) defining groups of observations.
* `k` : Nb. test observations, or nb. test groups if `group` is used, returned in each validation segment.
Keyword arguments: 
* `rep` : Nb. replications of the sampling.
* `seed` : Eventual seed for the `Random.MersenneTwister` generator. 
    
For each replication, the function builds a test set that can be used to validate a model. 

If `group` is used (must be a vector of length `n`), the function samples groups of observations instead of single 
observations. Such a group-sampling is required when the data are structured by groups and when the response to 
predict is correlated within groups. This prevents underestimation of the generalization error.

The function returns a list (vector) of `rep` elements. Each element of the list is a vector of the indexes 
(positions within 1:`n`) of the sampled observations.  

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
segmts(n, k; rep, seed = 123)

n = 10 
group = ["A", "B", "C", "D", "E", "A", "B", "C", "D", "E"]    # groups of the observations
tab(group)  
k = 2 ; rep = 4 
segm = segmts(group, k; rep)
#segm = segmts(group, k; rep seeed = 1234)
i = 1 
segm[i]
segm[i][1]
group[segm[i][1]]
```
""" 
function segmts(n::Signed, k::Signed; rep = 1, seed::Union{Nothing, Int} = nothing)
    Q = Vector{Int}
    s = list(Vector{Q}, rep)
    if isnothing(seed)
        vseed = [nothing for i in eachindex(s)]
    else 
        vseed = [seed + i - 1 for i in eachindex(s)]
    end
    for i in eachindex(s)
        s[i] = list(Q, 1)
        s[i][1] = StatsBase.sample(MersenneTwister(vseed[i]), 1:n, k; replace = false, ordered = true)
    end
    s
end

function segmts(group::Vector, k::Signed; rep = 1, seed::Union{Nothing, Int} = nothing)
    group = vec(group)
    Q = Vector{Int}
    s = list(Vector{Q}, rep)
    if isnothing(seed)
        vseed = [nothing for i in eachindex(s)]
    else 
        vseed = [seed + i - 1 for i in eachindex(s)]
    end
    yagg = unique(group)
    nlev = length(yagg)
    k = min(k, nlev)
    @inbounds for i = 1:rep
        s[i] = list(Q, 1)
        s[i][1] = StatsBase.sample(MersenneTwister(vseed[i]), 1:nlev, k; replace = false, ordered = true)
    end
    zs = copy(s)
    @inbounds for i = 1:rep
        u = s[i][1]
        v = findall(in(yagg[u]).(group))        # which(group %in% yagg[u])
        zs[i][1] = v
    end
    zs    
end

