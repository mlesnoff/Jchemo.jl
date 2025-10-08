"""
    fweight(X, v)
    fweight!(X::AbstractMatrix, v)
Weight each row of a matrix.
* `X` : Data (n, p).
* `v` : A weighting vector (n).

## Examples
```julia
using Jchemo, LinearAlgebra

X = rand(5, 2) 
w = rand(5) 
fweight(X, w)
diagm(w) * X

fweight!(X, w)
X
```
""" 
function fweight(X, v)
    X = ensure_mat(X)
    n, p = size(X)
    zX = similar(v, n, p)
    @inbounds for j = 1:p, i = 1:n
        zX[i, j] = X[i, j] * v[i]
    end  
    zX
end

function fweight!(X::AbstractMatrix, v)
    n, p = size(X)
    @inbounds for j = 1:p, i = 1:n
        X[i, j] *= v[i]
    end
end
