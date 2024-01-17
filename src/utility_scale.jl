"""
    fcenter(X, v)
    fcenter!(X::AbstractMatrix, v)
Center each column of `X`.
* `X` : Data.
* `v` : Centering vector.

## examples
```julia
n, p = 5, 6
X = rand(n, p)
xmeans = colmean(X)
fcenter(X, xmeans)
```
""" 
fcenter(X, v) = X .- vec(v)'

fcenter!(X::AbstractMatrix, v) = X .-= vec(v)'

"""
    fscale(X, v)
    fscale!(X::AbstractMatrix, v)
Scale each column of `X`.
* `X` : Data.
* `v` : Scaling vector.

## Examples
```julia
X = rand(5, 2) 
fscale(X, colstd(X))
```
""" 
fscale(X, v) = X ./ vec(v)'

fscale!(X::AbstractMatrix, v) = X ./= vec(v)'

"""
    fcscale(X, u, v)
    fcscale!(X, u, v)
Center and fscale each column of `X`.
* `X` : Data.
* `u` : Centering vector.
* `v` : Scaling vector.

## examples
```julia
n, p = 5, 6
X = rand(n, p)
xmeans = colmean(X)
xstds = colstd(X)
fcscale(X, xmeans, xstds)
```
""" 
function fcscale(X, u, v)
    zX = copy(ensure_mat(X))
    fcscale!(zX, u, v)
    zX
end

fcscale!(X::AbstractMatrix, u, v) = fscale!(fcenter!(X, u), v)
