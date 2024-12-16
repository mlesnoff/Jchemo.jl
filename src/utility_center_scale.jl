"""
    fcenter(X, v)
    fcenter!(X::AbstractMatrix, v)
Center each column of `X`.
* `X` : Data.
* `v` : Centering vector.

## examples
```julia
using Jchemo

n, p = 5, 6
X = rand(n, p)
xmeans = colmean(X)
fcenter(X, xmeans)
```
""" 
function fcenter(X, v)
    X = ensure_mat(X)
    p = nco(X)
    zX = similar(X)
    @Threads.threads for j = 1:p
        zX[:, j] .= vcol(X, j) .- v[j]
    end
    zX
end

function fcenter!(X::AbstractMatrix, v)
    X = ensure_mat(X)
    n, p = size(X)
    @inbounds for j = 1:p, i = 1:n
        X[i, j] -= v[j]
    end  
end

"""
    fscale(X, v)
    fscale!(X::AbstractMatrix, v)
Scale each column of `X`.
* `X` : Data.
* `v` : Scaling vector.

## Examples
```julia
using Jchemo

X = rand(5, 2) 
fscale(X, colstd(X))
```
""" 
function fscale(X, v)
    X = ensure_mat(X)
    p = nco(X)
    zX = similar(X)
    @Threads.threads for j = 1:p
        zX[:, j] .= vcol(X, j) ./ v[j]
    end
    zX
end

function fscale!(X::AbstractMatrix, v)
    X = ensure_mat(X)
    n, p = size(X)
    @inbounds for j = 1:p, i = 1:n
        X[i, j] /= v[j]
    end  
end

"""
    fcscale(X, u, v)
    fcscale!(X, u, v)
Center and fscale each column of `X`.
* `X` : Data.
* `u` : Centering vector.
* `v` : Scaling vector.

## examples
```julia
using Jchemo

n, p = 5, 6
X = rand(n, p)
xmeans = colmean(X)
xscales = colstd(X)
fcscale(X, xmeans, xscales)
```
""" 
function fcscale(X, u, v)
    X = ensure_mat(X)
    p = nco(X)
    zX = similar(X)
    @Threads.threads for j = 1:p
        zX[:, j] .= (vcol(X, j) .- u[j]) ./ v[j]
    end
    zX
end

function fcscale!(X::AbstractMatrix, u, v)
    X = ensure_mat(X)
    n, p = size(X)
    @inbounds for j = 1:p, i = 1:n
        X[i, j] = (X[i, j] - u[j]) / v[j]
    end  
end

