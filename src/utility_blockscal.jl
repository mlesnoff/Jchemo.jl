"""
    fblockscal(Xbl; bscales)
    fblockscal_frob(Xbl)
    fblockscal_frob(Xbl, weights = ones(nro(X[1]))
    fblockscal_mfa(Xbl, weights = ones(nro(X[1]))
    fblockscal_ncol(Xbl)
    fblockscal_sd(Xbl, weights = ones(nro(X[1]))
Scale a list of blocks (matrices).
* `Xbl` : List (vector) of blocks (matrices) of X-data. 
* `weights` : Weights of the observations (rows of the blocks). 
* `bscales` : A vector (of length equal to the nb. blocks) of the scalars diving the blocks.

Specificities of each function:
* `fblockscal`: Each block X is tranformed to X / `bscales'.
* `fblockscal_frob`: Let D be the diagonal matrix of vector `weights`,
    standardized to sum to 1. Each block X is divided by its Frobenius norm 
    = sqrt(tr(X' * D * X)). After this scaling, tr(X' * D * X) = 1.
* `fblockscal_mfa`: Each block X is divided by sqrt(lamda),
    where lambda is the dominant eigenvalue of X (this is the "MFA" approach).
* `fblockscal_ncol`: Each block X is divided by the nb. columns of the block.
* `fblockscal_sd`: Each block X is divided by sqrt(sum(weighted variances of the block-columns)).
    After this scaling, sum(weighted variances of the block-columns) = 1.

The functions return the scaled blocks and the scaling values.

## Examples
```julia
n = 5 ; p = 10 
X = rand(n, p) 
Xnew = X[1:3, :]

listbl = [3:4, 1, [6; 8:10]]
Xbl = mblock(X, listbl) 

Xbl[1]
Xbl[2]
Xbl[3]

bscales = ones(3)
res = fblockscal(Xbl, bscales) ;
res.bscales
res.X[3]
Xbl[3]

w = ones(n)
#w = mweight(rand(n))
D = Diagonal(mweight(w))
res = fblockscal_frob(Xbl, w) ;
res.bscales
k = 3 ; tr(Xbl[k]' * D * Xbl[3])^.5
tr(res.X[k]' * D * res.X[k])^.5

w = ones(n)
#w = mweight(rand(n))
res = fblockscal_mfa(Xbl, w) ;
res.bscales
k = 3 ; pcasvd(Xbl[k], w; nlv = 1).sv[1]

res = fblockscal_ncol(Xbl) ;
res.bscales
res.X[3]
Xbl[3] / size(Xbl[3], 2)

w = ones(n)
#w = mweight(rand(n))
res = fblockscal_sd(Xbl, w) ;
res.bscales
sum(colvar(res.X[3], w))

# To concatenate the returned blocks

X_concat = reduce(hcat, res.X)
```
"""
function fblockscal(Xbl, bscales)
    X = copy(Xbl)
    nbl = length(X)
    #Threads not faster
    @inbounds for k = 1:nbl
        X[k] = X[k] / bscales[k]
    end
    (Xbl = X, bscales)
end 

function fblockscal_frob(Xbl, weights::Weight)
    nbl = length(Xbl)
    bscales = list(eltype(Xbl[1]), nbl)
    @inbounds for k = 1:nbl
        bscales[k] =  frob(Xbl[k], weights)
    end
    fblockscal(Xbl, bscales)
end 

function fblockscal_mfa(Xbl, weights::Weight)
    nbl = length(Xbl)
    sqrtD = Diagonal(sqrt.(weights.w))
    bscales = list(eltype(Xbl[1]), nbl)
    @inbounds for k = 1:nbl
        xmeans = colmean(Xbl[k], weights)
        zX = fcenter(Xbl[k], xmeans)
        bscales[k] = nipals(sqrtD * zX).sv
    end
    fblockscal(Xbl, bscales)
end 

function fblockscal_ncol(Xbl)
    nbl = length(Xbl)
    bscales = list(eltype(Xbl[1]), nbl)
    @inbounds for k = 1:nbl
        bscales[k] =  nco(Xbl[k]) 
    end
    fblockscal(Xbl, bscales)
end 

function fblockscal_sd(Xbl, weights::Weight)
    nbl = length(Xbl)
    bscales = list(eltype(Xbl[1]), nbl)
    @inbounds for k = 1:nbl
        bscales[k] = sqrt(sum(colvar(Xbl[k], weights)))
    end
    fblockscal(Xbl, bscales)
end 


