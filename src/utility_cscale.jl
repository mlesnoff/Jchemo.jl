"""
    center(X, v)
    center!(X::AbstractMatrix, v)
Center each column of `X`.
* `X` : Data.
* `v` : Centering factors.

## examples
```julia
n, p = 5, 6
X = rand(n, p)
xmeans = colmean(X)
center(X, xmeans)
```
""" 
center(X, v) = X .- vec(v)'

function center!(X::AbstractMatrix, v)
    X .-= vec(v)'
end

"""
    scale(X, v)
    scale!(X::AbstractMatrix, v)
Scale each column of `X`.
* `X` : Data.
* `v` : Scaling factors.

## Examples
```julia
X = rand(5, 2) 
scale(X, colstd(X))
```
""" 
scale(X, v) = X ./ vec(v)'

function scale!(X::AbstractMatrix, v)
    X ./= vec(v)'
end

"""
    cscale(X, u, v)
    cscale!(X, u, v)
Center and scale each column of `X`.
* `X` : Data.
* `u` : Centering factors.
* `v` : Scaling factors.

## examples
```julia
n, p = 5, 6
X = rand(n, p)
xmeans = colmean(X)
xstds = colstd(X)
cscale(X, xmeans, xstds)
```
""" 
function cscale(X, u, v)
    zX = copy(ensure_mat(X))
    cscale!(zX, u, v)
    zX
end

cscale!(X::AbstractMatrix, u, v) = scale!(center!(X, u), v)


