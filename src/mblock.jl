"""
    mblock(X, listbl)
Make blocks from a matrix.
* `X` : X-data.
* `listbl` : A vector whose each component defines the colum numbers
    defining a block in `X`. The length of `listbl` is the number
    of blocks.

The function returns a list (vector) of blocks.

## Examples
```julia
n = 5 ; p = 10 
X = rand(n, p) 
listbl = [3:4, 1, [6; 8:10]]

X_bl = mblock(X, listbl)
X_bl[1]
X_bl[2]
X_bl[3]
```
"""
function mblock(X, listbl)
    nbl = length(listbl)
    zX = list(nbl, Matrix{Float64})
    @inbounds for i = 1:nbl
        zX[i] = ensure_mat(X[:, listbl[i]])
    end
    zX
end 
