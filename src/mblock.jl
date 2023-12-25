"""
    mblock(X, listbl)
Make blocks from a matrix.
* `X` : X-data.
* `listbl` : A vector whose each component defines 
    the colum numbers defining a block in `X`.
    The length of `listbl` is the number of blocks.

The function returns a list (vector) of blocks.

## Examples
```julia
n = 5 ; p = 10 
X = rand(n, p) 
listbl = [3:4, 1, [6; 8:10]]

Xbl = mblock(X, listbl)
Xbl[1]
Xbl[2]
Xbl[3]
```
"""
function mblock(X, listbl)
    Q = eltype(X[1, 1])
    nbl = length(listbl)
    Xbl = list(Matrix{Q}, nbl)
    @inbounds for i = 1:nbl
        Xbl[i] = ensure_mat(X[:, listbl[i]])
    end
    Xbl
end 
