"""
    segmkf(n, K; rep = 1)
    segmkf(n, K, group; rep = 1)
Build segments for K-fold cross-validation.  
* `n` : Total nb. observations in the dataset. The sampling 
    is implemented with 1:n.
* `K` : Nb. folds (segments) splitting the data. 
* `group` : A vector (n) defining blocks.
* `rep` : Nb. replications of the sampling.

Build K segments splitting the data that can be used to validate a model. 
The sampling can be replicated (`rep`).

If `group` is used (must be a vector of length n), the function samples entire 
blocks of observations instead of observations. Such a block-sampling is required 
when data is structured by blocks and when the response to predict is 
correlated within blocks. It prevents underestimation of the generalization error.

The function returns a list (vector) of `rep` elements. 
Each element of the list contains `K` segments (= `K` vectors).
Each segment contains the indexes (position within 1:`n`) of the sampled observations.    

## Examples
```julia
n = 10 ; K = 3 ; rep = 4 
segm = segmkf(n, K; rep) 
i = 1 
segm[i] # = replication "i"

# Block-sampling

n = 11 
group = ["A", "B", "C", "D", "E", "A", "B", "C", "D", "E", "A"]    # = blocks of the observations
unique(group)   
K = 3 ; rep = 4 
segm = segmkf(n, K, group; rep = rep)
i = 1 
segm[i]
group[segm[i][1]]
group[segm[i][2]]
group[segm[i][3]]
```
""" 
function segmkf(n, K; rep = 1)
    m = K - n % K ;
    s = list(rep)
    @inbounds for i = 1:rep
        s[i] = list(K, Vector{Int64})
        v = [randperm(n) ; repeat([0], outer = m)] 
        v = reshape(v, K, :)
        @inbounds for j = 1:K 
            s[i][j] = sort(filter(x -> x > 0, v[j, :]))
        end
    end
    s
end

function segmkf(n, K, group; rep = 1)
    # n is not used but is kept for multiple dispatch
    group = vec(group) # must be of length n
    s = list(rep)
    yagg = unique(group)
    zn = length(yagg)
    K = min(K, zn)
    m = K - zn % K ;
    s = list(rep)
    @inbounds for i = 1:rep
        s[i] = list(K, Vector{Int64})
        v = [randperm(zn) ; repeat([0], outer = m)] 
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

"""
    segmts(n, m; rep = 1, seed = nothing)
    segmts(n, m, group; rep = 1, seed = nothing)
Build segments for "test-set" validation.
* `n` : Total nb. observations in the dataset. The sampling 
    is implemented within 1:`n`.
* `m` : Nb. observations, or groups (if `group` is used), returned in each segment.
* `seed` : Can set the seet for the `Random.MersenneTwister` generator. 
    Must be of length = `rep`. When `nothing`, the seed is random at each sampling.
* `group` : A vector (n) defining blocks of observations.
* `rep` : Nb. replications of the sampling.
    
Build a test set that can be used to validate a model.
The sampling can be replicated (`rep`).

If `group` is used (must be a vector of length n), the function samples entire 
groups (= blocks) of observations instead of observations. 
Such a block-sampling is required when data is structured by blocks and 
when the response to predict is correlated within blocks. 
This prevents underestimation of the generalization error.

The function returns a list (vector) of `rep` elements. 
Each element of the list contains `K` segments (= `K` vectors).
Each segment contains the indexes (position within 1:`n`) of the sampled observations.  

## Examples
```julia
n = 10 ; m = 3 ; rep = 4 
segm = segmts(n, m; rep) 
i = 1 
segm[i]

segmts(10, 4; seed = 3)
segmts(10, 4; rep = 2, seed = [1 ; 3])

# Block-sampling

n = 11
group = ["A", "B", "C", "D", "E", "A", "B", "C", "D", "E", "A"]    # = blocks of the observations
tab(group)
m = 2 ; rep = 4 
segm = segmts(n, m, group; rep)
i = 1 
segm[i]
segm[i][1]
group[segm[i][1]]
```
""" 
function segmts(n, m; rep = 1, seed = nothing)
    s = list(rep)
    for i = 1:rep
        s[i] = list(1, Vector{Int64})
        if isnothing(seed)
            s[i][1] = sample(1:n, m; replace = false, ordered = true)
        else
            s[i][1] = sample(MersenneTwister(Int64(seed[i])), 1:n, m;
                replace = false, ordered = true)
        end
    end
    s
end

function segmts(n, m, group; rep = 1, seed = nothing)
    # n is not used but is kept for multiple dispatch
    group = vec(group) # must be of length n
    s = list(rep)
    yagg = unique(group)
    zn = length(yagg)
    m = min(m, zn)
    @inbounds for i = 1:rep
        s[i] = list(1, Vector{Int64})
        if isnothing(seed)
            s[i][1] = sample(1:zn, m; replace = false, ordered = true)
        else
            s[i][1] = sample(MersenneTwister(Int64(seed[i])), 1:zn, m; 
                replace = false, ordered = true)
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

