"""
    segmkf(n::Int, K::Int; rep = 1)
    segmkf(group::Vector, K::Int; rep = 1)
Build segments for K-fold cross-validation.  
* `n` : Total nb. observations in the dataset. The sampling 
    is implemented with 1:n.
* `K` : Nb. folds (segments) splitting the data. 
* `group` : A vector (n) defining blocks.
* `rep` : Nb. replications of the sampling.

Build the `n` observations to K segments that can be used for 
K-fold cross-validation. The sampling can be replicated (`rep`).

If `group` is used (vector of length n), the function samples entire 
blocks of observations instead of observations. Such a block-sampling is required 
when data is structured by blocks and when the response to predict is 
correlated within blocks. It prevents underestimation of the generalization error.

The function returns a list (vector) of `rep` elements. 
Each element of the list contains `K` segments (= `K` vectors).
Each segment contains the indexes (position within 1:`n`) of the sampled 
observations.    

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
function segmkf(n::Int, K::Int; rep = 1)
    m = K - n % K ;
    Q = Vector{Int}
    s = list(rep, Vector{Q})
    @inbounds for i = 1:rep
        s[i] = list(K, Q)
        v = [randperm(n) ; repeat([0], outer = m)] 
        v = reshape(v, K, :)
        @inbounds for j = 1:K 
            s[i][j] = sort(filter(x -> x > 0, v[j, :]))
        end
    end
    s
end

function segmkf(group::Vector, K::Int; rep = 1)
    yagg = unique(group)
    nlev = length(yagg)
    K = min(K, nlev)
    m = K - nlev % K ;
    Q = Vector{Int}
    s = list(rep, Vector{Q})
    @inbounds for i = 1:rep
        s[i] = list(K, Q)
        v = [randperm(nlev) ; repeat([0], outer = m)] 
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

