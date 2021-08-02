"""
    segmts(n; m, nrep = 1)
Build segments for "test-set" validation.  
    * `n` : Total nb. observations in the dataset. The sampling 
        is implemented within 1:n.
    * `m` : Nb. observations in each segment. 
    * `nrep` : Nb. replications of the sampling.
    
This builds a test set (eventually with replications) that can be used to validate a model.

The function returns a list (Vector::Any) of nrep elements. 
Each element of the list contains a vector (= segment) 
of the indexes (position within 1:n) of the m sampled observations.

## Examples
```julia
n = 10 ; m = 3 ; nrep = 4 ;
z = segmts(n; m, nrep) 
i = 1 ;
z[i]
z[i][1]
```
""" 
function segmts(n; m, nrep = 1)
    s = list(nrep)
    for i = 1:nrep
        s[i] = list(1)
        s[i][1] = sample(1:n, m, replace = false, ordered = true)
    end
    s
end

"""
    segmts(n, y; m, nrep = 1)
Build segments (with block-sampling) for "test-set" validation.  
    * `n` : Total nb. observations in the dataset. The sampling 
        is implemented with 1:n.
    * `y` : A vector (n,) defining the blocks.
    * `m` : Nb. blocks in the segment. 
    * `nrep` : Nb. replications of the sampling.

Build a test set (eventually with replications) that can be used to validate a model.

The function samples entire blocks of observations instead of observations. 
Vector `y` (defining the blocks) must be of length n.     

Such a block-sampling is required when data is structured by blocks and 
when the response to predict is correlated within blocks.   
It prevents high underestimation of the generalization error.

The function returns a list (Vector::Any) of nrep elements. 
Each element of the list contains a vector (= segment) 
of the indexes (position within 1:n) of the sampled observations.
Each segment contains m blocks.

## Examples
```julia
n = 10 ;
y = ["A", "B", "C", "D", "E", "A", "B", "C", "D", "E"]    # The blocks of the observations
unique(y)    # Print of the blocks
m = 2 ; nrep = 4 ;
z = segmts(n, y; m, nrep)
i = 1 ;
z[i]
z[i][1]
y[z[i][1]]
```
""" 
function segmts(n, y; m, nrep = 1)
    y = vec(y) # must be size (n,)
    s = list(nrep)
    yagg = unique(y)
    zn = length(yagg)
    m = min(m, zn)
    @inbounds for i = 1:nrep
        s[i] = list(1)
        s[i][1] = sample(1:zn, m, replace = false, ordered = true)
    end
    zs = copy(s)
    @inbounds for i = 1:nrep
        u = s[i][1]
        v = findall(in(yagg[u]).(y))        # which(y %in% yagg[u])
        zs[i][1] = v
    end
    zs    
end

"""
    segmkf(n; K, nrep = 1)
Build segments for K-fold cross-validation.  
    * `n` : Total nb. observations in the dataset. The sampling 
        is implemented with 1:n.
    * `K` : Nb. folds (segments) splitting the data. 
    * `nrep` : Nb. replications of the sampling.

Build K segments splitting the data (eventually with replications) that can 
be used to validate a model.

The function returns a list (Vector::Any) of nrep elements. 
Each element of the list contains K vectors (= K segments).
Each segment contains the indexes (position within 1:n) of the sampled observations.    

## Examples
```julia
n = 10 ; m = 3 ; nrep = 4 ;
z = segmts(n; m, nrep) 
i = 1 ;
z[i]
z[i][1]
```
""" 
function segmkf(n; K, nrep = 1)
    m = K - n % K ;
    s = list(nrep)
    @inbounds for i = 1:nrep
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
    segmkf(n, y; K, nrep = 1)
Build segments (with block-sampling) for K-fold cross-validation.  
    * `n` : Total nb. observations in the dataset. The sampling 
        is implemented with 1:n.
    * `y` : A vector (n,) defining the blocks.
    * `K` : Nb. folds (segments) splitting the data. 
    * `nrep` : Nb. replications of the sampling.

Build K segments splitting the data (eventually with replications) that can 
be used to validate a model.

The function samples entire blocks of observations instead of observations. 
Vector `y` (defining the blocks) must be of length n.     
    
Such a block-sampling is required when data is structured by blocks and 
when the response to predict is correlated within blocks.   
It prevents high underestimation of the generalization error.

The function returns a list (Vector::Any) of nrep elements. 
Each element of the list contains K vectors (= K segments).
Each segment contains the indexes (position within 1:n) of the sampled observations.    

## Examples
```julia
n = 10 ;
y = ["A", "B", "C", "D", "E", "A", "B", "C", "D", "E"]    # The blocks of the observations
unique(y)    # Print of the blocks
K = 3 ; nrep = 4 ;
z = segmkf(n, y; K, nrep)
i = 1 ;
z[i]
z[i][1]
y[z[i][1]]
y[z[i][2]]
y[z[i][3]]
```
""" 
function segmkf(n, y; K, nrep = 1)
    y = vec(y) # must be size (n,)
    s = list(nrep)
    yagg = unique(y)
    zn = length(yagg)
    K = min(K, zn)
    m = K - zn % K ;
    s = list(nrep)
    @inbounds for i = 1:nrep
        s[i] = list(K)
        v = [randperm(zn) ; repeat([0], outer = m)] 
        v = reshape(v, K, :)
        @inbounds for j = 1:K 
            s[i][j] = sort(filter(x -> x > 0, v[j, :]))
        end
    end
    zs = copy(s)
    @inbounds for i = 1:nrep
        @inbounds for j = 1:K
            u = s[i][j]
            v = findall(in(yagg[u]).(y))       
            zs[i][j] = v
        end
    end
    zs    
end




