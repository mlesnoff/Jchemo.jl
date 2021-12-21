"""
    mblocks(X, listbl)
Make blocks from a matrix.
* `X` : X-data.
* `listbl` : A vector whose each component defines the colum numbers
    defining a block in `X` . The length of `listbl` is the number
    of blocks.

The function returns a list of blocks.

## Examples
```julia
n = 5 ; p = 10 
X = rand(n, p) 
listbl = [3:4, 1, [6; 8:10]]

X_bl = mblocks(X, listbl)
```
"""
function mblocks(X, listbl)
    nbl = length(listbl)
    zX = list(nbl, Matrix{Float64})
    @inbounds for i = 1:nbl
        zX[i] = ensure_mat(X[:, listbl[i]])
    end
    zX
end 

"""
    blockscal(X, weights = ones(size(X, 1)); listbl, scal = nothing)
Autoscale blocks of a matrix.
* `X` : List (vector) of blocks (matrices) of X-data. 
    Each component of the list is a block.
* `weights` : Weights of the observations (rows). Only used if `scal = nothing`.
* `scal` : If `nothing`, each block is autoscaled (i.e. divided) by the 
    the square root of the sum of the variances of each column of the block.
    Else, `scal` must be a vector of the scaling values dividing the blocks.

The function returns the scaled blocks, and the scaling values.

Vector `weights` is internally normalized to sum to 1.

## Examples
```julia
n = 5 ; p = 10 
X = rand(n, p) 
listbl = [3:4, 1, [6; 8:10]]
X_bl = mblocks(X, listbl)

scal = nothing
#scal = [3.1 2 .7]
res = blockscal(X_bl; scal = scal) ;
res.X
res.scal
sum(colvars(res.X[1]))

X_concat = reduce(hcat, res.X)
```
"""
function blockscal(X, weights = ones(size(X[1], 1)); scal = nothing)
    weights = mweights(weights)
    nbl = length(X)
    isnothing(scal) ? zscal = similar(X[1], nbl) : zscal = copy(scal)
    X_sc = list(nbl, Matrix{Float64})
    @inbounds for i = 1:nbl
        if isnothing(scal)
            zscal[i] = sqrt(sum(colvars(X[i], weights)))
        end
        X_sc[i] = X[i] / zscal[i]
    end
    (X = X_sc, scal = zscal)
end 

