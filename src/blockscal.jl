"""
    mblocks(X, listbl)
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
    blockscal(X_bl; scal)
    blockscal_frob(X_bl, weights = ones(size(X[1], 1)))
    blockscal_ncol(X_bl)
    blockscal_sd(X_bl, weights = ones(size(X[1], 1)))
Scale a list of blocks (matrices).
* `X_bl` : List (vector) of blocks (matrices) of X-data. 
* `weights` : Weights of the observations (rows of the blocks). 
* `scal` : A vector (of length equal to the nb. blocks) of the scalars diving the blocks.

Specificities:
* `blockscal`: Each block X is tranformed to X / scalar.
* `blockscal_frob`: Let us note D the diagonal matrix of vector `weights`.
Each block X is divided by its Frobenius norm = sqrt(trace(X' * D * X)).
After scaling, trace(X' * D * X) = 1.
* `blockscal_ncol`: Each block X is tranformed to X / nb. columns of the block.
* `blockscal_sd`: Each block X is transformed to X / sqrt(sum(weighted variances of the block-columns)).
After scaling, sum(weighted variances of the block-columns) = 1.

The functions return the scaled blocks and the scaling values.

**Note:** In `blockscal_sd`, for the true weighted variances, vector `weights` must be preliminary 
normalized to sum to 1 (`weights` is not internally normalized).

## Examples
```julia
n = 5 ; p = 10 
X = rand(n, p) 
listbl = [3:4, 1, [6; 8:10]]
X_bl = mblocks(X, listbl)
i = 3
X_bl[i]

scal = [3.1 ; 2 ; .7]
res = blockscal(X_bl; scal = scal) ;
res.scal

w = mweights(ones(n))
#w = mweights(collect(1:n))
res = blockscal_sd(X_bl, w) ;
res.scal
i = 3 ; sum(colvars(res.X[i], w))

w = ones(n)
#w = collect(1:n)
D = Diagonal(w)
res = blockscal_frob(X_bl, w) ;
res.scal
i = 3 ; tr(X_bl[i]' * D * X_bl[3])^.5
tr(res.X[i]' * D * res.X[i])^.5

# To concatenate the returned blocks

X_concat = reduce(hcat, res.X)
```
"""
function blockscal(X_bl; scal)
    X = copy(X_bl)
    nbl = length(X)
    @inbounds for i = 1:nbl
        X[i] = X[i] / scal[i]
    end
    (X = X, scal)
end 

function blockscal_frob(X_bl, weights = ones(size(X[1], 1)))
    nbl = length(X_bl)
    scal = list(nbl, Float64)
    @inbounds for i = 1:nbl
        scal[i] =  sqrt(sum(colnorms2(X_bl[i], weights)))
    end
    blockscal(X_bl; scal = scal)
end 

function blockscal_ncol(X_bl)
    nbl = length(X_bl)
    scal = list(nbl, Float64)
    @inbounds for i = 1:nbl
        scal[i] =  size(X_bl[i], 2) 
    end
    blockscal(X_bl; scal = scal)
end 

function blockscal_sd(X_bl, weights = ones(size(X[1], 1)))
    nbl = length(X_bl)
    scal = list(nbl, Float64)
    @inbounds for i = 1:nbl
        scal[i] = sqrt(sum(colvars(X_bl[i], weights)))
    end
    blockscal(X_bl; scal = scal)
end 


