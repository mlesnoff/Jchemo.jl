"""
    fcenter(X, v)
    fcenter!(X::AbstractMatrix, v)
Center each column of a matrix.
* `X` : Data (n, p).
* `v` : Centering vector (p).

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
    n, p = size(X)
    zX = similar(X)
    @inbounds for j = 1:p, i = 1:n
        zX[i, j] = X[i, j] - v[j]
    end  
    zX
end

function fcenter!(X::AbstractMatrix, v)
    n, p = size(X)
    @inbounds for j = 1:p, i = 1:n
        X[i, j] -= v[j]
    end  
end

"""
    fscale(X, v)
    fscale!(X::AbstractMatrix, v)
Scale each column of a matrix.
* `X` : Data (n, p).
* `v` : Scaling vector (p).

## Examples
```julia
using Jchemo

X = rand(5, 2) 
fscale(X, colstd(X))
```
""" 
function fscale(X, v)
    X = ensure_mat(X)
    n, p = size(X)
    zX = similar(X)
    @inbounds for j = 1:p, i = 1:n
        zX[i, j] = X[i, j] / v[j]
    end  
    zX
end

function fscale!(X::AbstractMatrix, v)
    n, p = size(X)
    @inbounds for j = 1:p, i = 1:n
        X[i, j] /= v[j]
    end 
end

"""
    fcscale(X, u, v)
    fcscale!(X, u, v)
Center and scale each column of a matrix.
* `X` : Data  (n, p).
* `u` : Centering vector (p).
* `v` : Scaling vector (p).

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
    n, p = size(X)
    zX = similar(X)
    @inbounds for j = 1:p, i = 1:n
        zX[i, j] = (X[i, j] - u[j]) / v[j]
    end  
    zX
end

function fcscale!(X::AbstractMatrix, u, v)
    n, p = size(X)
    @inbounds for j = 1:p, i = 1:n
        X[i, j] = (X[i, j] - u[j]) / v[j]
    end  
end

