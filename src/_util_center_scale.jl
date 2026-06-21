"""
    fcenter(X, v) 
    fcenter!(X::Matrix{Q}, v::Vector{Q}) where Q <: AbstractFloat
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
    zX = copy(ensure_mat(X))
    fcenter!(zX, v)
    zX
end

function fcenter!(X::Matrix{Q}, v::Vector{Q}) where Q <: AbstractFloat
    @inbounds for j in axes(X, 2), i in axes(X, 1)
        X[i, j] -= v[j]
    end  
end

"""
    fscale(X, v)
    fscale!(X::Matrix{Q}, v::Vector{Q}) where Q <: AbstractFloat
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
    zX = copy(ensure_mat(X))
    fscale!(zX, v)
    zX
end

function fscale!(X::Matrix{Q}, v::Vector{Q}) where Q <: AbstractFloat
    @inbounds for j in axes(X, 2), i in axes(X, 1)
        X[i, j] /= v[j]
    end 
end

"""
    fcscale(X, u, v)
    fcscale!(X::Matrix{Q}, u::Vector{Q}, v::Vector{Q}) where Q <: AbstractFloat
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
    zX = copy(ensure_mat(X))
    fcscale!(zX, u, v)
    zX
end

function fcscale!(X::Matrix{Q}, u::Vector{Q}, v::Vector{Q}) where Q <: AbstractFloat
    @inbounds for j in axes(X, 2), i in axes(X, 1)
        X[i, j] = (X[i, j] - u[j]) / v[j]
    end  
end



