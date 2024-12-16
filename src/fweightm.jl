"""
    fweightm(X, v)
    fweightm!(X::AbstractMatrix, v)
Weight each row of a matrix.
* `X` : Data (n, p).
* `v` : A weighting vector (n).

## Examples
```julia
using Jchemo, LinearAlgebra

X = rand(5, 2) 
w = rand(5) 
fweightm(X, w)
diagm(w) * X

fweightm!(X, w)
X
```
""" 
function fweightm(X, v)
    X = ensure_mat(X)
    n, p = size(X)
    zX = similar(X)
    @inbounds for j = 1:p, i = 1:n
        zX[i, j] = X[i, j] * v[i]
    end  
    zX
end

function fweightm!(X::AbstractMatrix, v)
    n, p = size(X)
    @inbounds for j = 1:p, i = 1:n
        X[i, j] *= v[i]
    end
end
