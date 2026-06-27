"""
    segmkf(n::Signed, K::Signed; rep = 1, seed::Union{Nothing, Int} = nothing)
    segmkf(group::Vector, K::Signed; rep = 1, seed::Union{Nothing, Int} = nothing)
Build segments of observations for K-fold cross-validation.  
* `n` : Total nb. of observations in the dataset. The sampling is implemented within 1:`n`.
* `group` : A vector (`n`) defining blocks of observations.
* `K` : Nb. folds (segments) splitting the `n` observations. 
Keyword arguments:
* `rep` : Nb. replications of the K-fold sampling.
* `seed` : Eventual seed for the `Random.MersenneTwister` generator. 

For each replication, the function splits the `n` observations to `K` segments  that can be used for K-fold 
cross-validation. 

If `group` is used (must be a vector of length `n`), the function samples entire groups (= blocks) of observations 
instead of observations. Such a block-sampling is required when data is structured by blocks and when the response to 
predict is correlated within blocks. This prevents underestimation of the generalization error.

The function returns a list (vector) of `rep` elements. Each element of the list contains `K` segments (= `K` vectors). 
Each segment contains the indexes (position within 1:`n`) of the sampled observations.    

## Examples
```julia
using Jchemo 

n = 10 ; K = 3
rep = 4 
segm = segmkf(n, K; rep)
#segm = segmkf(n, K; rep, seed = 1234)
i = 1 
segm[i]
segm[i][1]

n = 10 
group = ["A", "B", "C", "D", "E", "A", "B", "C", "D", "E"]    # blocks of the observations
tab(group) 
K = 3 ; rep = 4 
segm = segmkf(group, K; rep)
i = 1 
segm[i]
segm[i][1]
group[segm[i][1]]
group[segm[i][2]]
group[segm[i][3]]
```
""" 
function segmkf(n::Signed, K::Signed; rep = 1, seed::Union{Nothing, Int} = nothing)
    Q = Vector{Int}
    m = K - n % K ;
    s = list(Vector{Q}, rep)
    if isnothing(seed)
        vseed = [nothing for i in eachindex(s)]
    else 
        vseed = [seed + i - 1 for i in eachindex(s)]
    end
    @inbounds for i = 1:rep
        s[i] = list(Q, K)
        v = [Random.randperm(MersenneTwister(vseed[i]), n) ; fill(0, m)] 
        v = reshape(v, K, :)
        @inbounds for j = 1:K 
            s[i][j] = sort(filter(x -> x > 0, v[j, :]))
        end
    end
    s
end

function segmkf(group::Vector, K::Signed; rep = 1, seed::Union{Nothing, Int} = nothing)
    Q = Vector{Int}
    s = list(Vector{Q}, rep)
    if isnothing(seed)
        vseed = [nothing for i in eachindex(s)]
    else 
        vseed = [seed + i - 1 for i in eachindex(s)]
    end
    yagg = unique(group)
    nlev = length(yagg)
    K = min(K, nlev)
    m = K - nlev % K ;
    @inbounds for i = 1:rep
        s[i] = list(Q, K)
        v = [Random.randperm(MersenneTwister(vseed[i]), nlev) ; fill(0, m)] 
        v = reshape(v, K, :)
        @inbounds for j = 1:K 
            s[i][j] = sort(filter(x -> x > 0, v[j, :]))
        end
    end
    zs = copy(s)
    @inbounds for i = 1:rep
        @inbounds for j = 1:K
            u = s[i][j]
            v = findall(in(yagg[u]).(group))       
            zs[i][j] = v
        end
    end
    zs    
end

