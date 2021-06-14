"""
```
    colmeans(X)
    colmeans(X, w)
```

Compute the mean of each column of X.
- X : Matrix (n, p) or vector (p,).
- w : Vector of weights (n,).
Return a vector.

**Note:** For a true weighted mean, w must be preliminary normalized to sum to 1.
""" 
colmeans(X) = vec(Statistics.mean(X; dims = 1))

colmeans(X, w) = vec(w' * ensure_mat(X))

"""
```
    colvars(X)
    colvars(X, w)
```

Compute the variance (uncorrected) of each column of X.
- X : Matrix (n, p) or vector (p,).
- w : Vector of weights (n,).
Return a vector.

**Note:** For a true weighted variance, w must be preliminary normalized to sum to 1.
""" 
colvars(X) = vec(Statistics.var(X; corrected = false, dims = 1))

function colvars(X, w)
    p = size(X, 2)
    z = colmeans(X, w)
    @inbounds for j = 1:p
        z[j] = dot(view(w, :), (vcol(X, j) .- z[j]).^2)        
    end
    z 
end

"""
```
    center(X, v) 
    center!(X, v)
```

Center each column of X.
- X : Matrix (n, p), or vector (n,)
- v : Centering vector (p,)
""" 
function center(X, v)
    M = copy(X)
    center!(M, v)
    M
end

function center!(X, v)
    p = size(X, 2)
    @inbounds for j = 1:p
        X[:, j] .= vcol(X, j) .- v[j]
    end
end

"""
```
scale(X, v)
scale!(X, v) 
```

Scale each column of X.
- X : Matrix (n, p), or vector (n,).
- v : Scaling vector (p,).
""" 

function scale(X, v)
    M = copy(X)
    scale!(M, v)
    M
end

function scale!(X, v)
    p = size(X, 2)
    @inbounds for j = 1:p
        X[:, j] .= vcol(X, j) ./ v[j]
    end
end



