"""
    segmkf(n, K; rep = 1)
Build segments for K-fold cross-validation.  
* `n` : Total nb. observations in the dataset. The sampling 
    is implemented with 1:n.
* `K` : Nb. folds (segments) splitting the data. 
* `rep` : Nb. replications of the sampling.

Build K segments splitting the data (eventually with replications) that can 
be used to validate a model.

The function returns a list (Vector::Any) of rep elements. 
Each element of the list contains K vectors (= K segments).
Each segment contains the indexes (position within 1:n) of the sampled observations.    

## Examples
```julia
n = 10 ; m = 3 ; rep = 4 ;
z = segmts(n, m; rep) 
i = 1 ;
z[i]
z[i][1]
```
""" 
function segmkf(n, K; rep = 1)
    m = K - n % K ;
    s = list(rep)
    @inbounds for i = 1:rep
        s[i] = list(K)
        v = [randperm(n) ; repeat([0], outer = m)] 
        v = reshape(v, K, :)
        @inbounds for j = 1:K 
            s[i][j] = sort(filter(x -> x > 0, v[j, :]))
        end
    end
    s
end

"""
    segmkf(n, K, group; rep = 1)
Build segments (with block-sampling) for K-fold cross-validation.  
* `n` : Total nb. observations in the dataset. The sampling 
    is implemented with 1:n.
* `group` : A vector (n,) defining the blocks.
* `K` : Nb. folds (segments) splitting the data. 
* `rep` : Nb. replications of the sampling.

Build K segments splitting the data (eventually with replications) that can 
be used to validate a model.

The function samples entire blocks of observations instead of observations. 
Vector `group` (defining the blocks) must be of length n.     
    
Such a block-sampling is required when data is structured by blocks and 
when the response to predict is correlated within blocks.   
It prevents high underestimation of the generalization error.

The function returns a list (Vector::Any) of rep elements. 
Each element of the list contains K vectors (= K segments).
Each segment contains the indexes (position within 1:n) of the sampled observations.    

## Examples
```julia
n = 10 ;
group = ["A", "B", "C", "D", "E", "A", "B", "C", "D", "E"]    # The blocks of the observations
unique(group)    # Print of the blocks
K = 3 ; rep = 4 ;
z = segmkf(n, K, group; rep)
i = 1 ;
z[i]
z[i][1]
group[z[i][1]]
group[z[i][2]]
group[z[i][3]]
```
""" 
function segmkf(n, K, group; rep = 1)
    group = vec(group) # must be size (n,)
    s = list(rep)
    yagg = unique(group)
    zn = length(yagg)
    K = min(K, zn)
    m = K - zn % K ;
    s = list(rep)
    @inbounds for i = 1:rep
        s[i] = list(K)
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
    segmts(n, m; rep = 1)
Build segments for "test-set" validation.
* `n` : Total nb. observations in the dataset. The sampling 
    is implemented within 1:n.
* `m` : Nb. observations in each segment.
* `rep` : Nb. replications of the sampling.
    
This builds a test set (eventually replicated) that can be used to validate a model.

The function returns a list (Vector::Any) of rep elements. 
Each element of the list contains a vector (= segment) 
of the indexes (position within 1:n) of the m sampled observations.

## Examples
```julia
n = 10 ; m = 3 ; rep = 4 ;
z = segmts(n, m; rep) 
i = 1 ;
z[i]
z[i][1]
```
""" 
function segmts(n, m; rep = 1)
    s = list(rep)
    for i = 1:rep
        s[i] = list(1)
        s[i][1] = sample(1:n, m, replace = false, ordered = true)
    end
    s
end

"""
    segmts(n, m, group; rep = 1)
Build segments with block-sampling for "test-set" validation.  
* `n` : Total nb. observations in the dataset. The sampling 
    is implemented with 1:n.
* `group` : A vector (n,) defining the blocks.
* `m` : Nb. blocks in the segment. 
* `rep` : Nb. replications of the sampling.

Build a test set (eventually replicated) that can be used to validate a model.

The function samples entire blocks of observations instead of observations. 
Vector `group` (defining the blocks) must be of length n.     

Such a block-sampling is required when data is structured by blocks and 
when the response to predict is correlated within blocks.   
It prevents high underestimation of the generalization error.

The function returns a list (Vector::Any) of rep elements. 
Each element of the list contains a vector (= segment) 
of the indexes (position within 1:n) of the sampled observations.
Each segment contains m blocks.

## Examples
```julia
n = 10 ;
group = ["A", "B", "C", "D", "E", "A", "B", "C", "D", "E"]    # The blocks of the observations
unique(group)    # Print of the blocks
m = 2 ; rep = 4 ;
z = segmts(n, m, group; rep)
i = 1 ;
z[i]
z[i][1]
group[z[i][1]]
```
""" 
function segmts(n, m, group; rep = 1)
    group = vec(group) # must be size (n,)
    s = list(rep)
    yagg = unique(group)
    zn = length(yagg)
    m = min(m, zn)
    @inbounds for i = 1:rep
        s[i] = list(1)
        s[i][1] = sample(1:zn, m, replace = false, ordered = true)
    end
    zs = copy(s)
    @inbounds for i = 1:rep
        u = s[i][1]
        v = findall(in(yagg[u]).(group))        # which(group %in% yagg[u])
        zs[i][1] = v
    end
    zs    
end






