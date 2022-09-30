"""
    blockscal(X_bl; bscales)
    blockscal_frob(X_bl)
    blockscal_frob(X_bl, weights = ones(size(X[1], 1)))
    blockscal_mfa(X_bl, weights = ones(size(X[1], 1)))
    blockscal_ncol(X_bl)
    blockscal_sd(X_bl, weights = ones(size(X[1], 1)))
Scale a list of blocks (matrices).
* `X_bl` : List (vector) of blocks (matrices) of X-data. 
* `weights` : Weights of the observations (rows of the blocks). 
* `bscales` : A vector (of length equal to the nb. blocks) of the scalars diving the blocks.

Specificities of each function:
* `blockscal`: Each block X is tranformed to X / `bscales'.
* `blockscal_frob`: Let D be the diagonal matrix of vector `weights`.
    Each block X is divided by its Frobenius norm = sqrt(trace(X' * D * X)).
    After this scaling, trace(X' * D * X) = 1.
* `blockscal_mfa`: Each block X is divided by sqrt(lamda),
    where lambda is the dominant eigenvalue of X (this is the "MFA" approach).
* `blockscal_ncol`: Each block X is divided by the nb. columns of the block.
* `blockscal_sd`: Each block X is divided by sqrt(sum(weighted variances of the block-columns)).
    After this scaling, sum(weighted variances of the block-columns) = 1.

The functions return the scaled blocks and the scaling values.

## Examples
```julia
n = 5 ; p = 10 
X = rand(n, p) 
Xnew = X[1:3, :]

listbl = [3:4, 1, [6; 8:10]]
X_bl = mblock(X, listbl) 

X_bl[1]
X_bl[2]
X_bl[3]

bscales = ones(3)
res = blockscal(X_bl, bscales) ;
res.bscales
res.X[3]
X_bl[3]

w = ones(n)
#w = collect(1:n)
D = Diagonal(mweight(w))
res = blockscal_frob(X_bl, w) ;
res.bscales
i = 3 ; tr(X_bl[i]' * D * X_bl[3])^.5
tr(res.X[i]' * D * res.X[i])^.5

w = ones(n)
#w = collect(1:n)
res = blockscal_mfa(X_bl, w) ;
res.bscales
i = 3 ; pcasvd(X_bl[i], w; nlv = 1).sv[1]

res = blockscal_ncol(X_bl) ;
res.bscales
res.X[3]
X_bl[3] / size(X_bl[3], 2)

w = ones(n)
#w = collect(1:n)
res = blockscal_sd(X_bl, w) ;
res.bscales
sum(colvar(res.X[3], w))

# To concatenate the returned blocks

X_concat = reduce(hcat, res.X)
```
"""
function blockscal(X_bl, 
        bscales)
    X = copy(X_bl)
    nbl = length(X)
    @inbounds for i = 1:nbl
        X[i] = X[i] / bscales[i]
    end
    (X = X, bscales)
end 

function blockscal_frob(X_bl, 
        weights = ones(size(X[1], 1)))
    nbl = length(X_bl)
    bscales = list(nbl, Float64)
    sqrtw = sqrt.(mweight(weights))
    sqrtD = Diagonal(sqrtw)
    @inbounds for i = 1:nbl
        bscales[i] =  sqrt(ssq(sqrtD * X_bl[i]))
    end
    blockscal(X_bl, bscales)
end 

function blockscal_mfa(X_bl, 
        weights = ones(size(X[1], 1)))
    nbl = length(X_bl)
    sqrtw = sqrt.(mweight(weights))
    sqrtD = Diagonal(sqrtw)
    bscales = list(nbl, Float64)
    @inbounds for k = 1:nbl
        xmeans = colmean(X_bl[k], weights)
        zX = center(X_bl[k], xmeans)
        bscales[k] = nipals(sqrtD * zX).sv
    end
    blockscal(X_bl, bscales)
end 

function blockscal_ncol(X_bl)
    nbl = length(X_bl)
    bscales = list(nbl, Float64)
    @inbounds for i = 1:nbl
        bscales[i] =  size(X_bl[i], 2) 
    end
    blockscal(X_bl, bscales)
end 

function blockscal_sd(X_bl,
        weights = ones(size(X[1], 1)))
    nbl = length(X_bl)
    bscales = list(nbl, Float64)
    @inbounds for i = 1:nbl
        bscales[i] = sqrt(sum(colvar(X_bl[i], weights)))
    end
    blockscal(X_bl, bscales)
end 


