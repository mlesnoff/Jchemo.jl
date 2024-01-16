"""
    fblockscal(Xbl; bscales)
    fblockscal!(Xbl::Vector, bscales::Vector)
    fblockscal_frob(Xbl, weights::Weight)
    fblockscal_mfa(Xbl, weights::Weight)
    fblockscal_ncol(Xbl)
    fblockscal_sd(Xbl, weights::Weight)
Scale multiblock X-data.
* `Xbl` : List of blocks (vector of matrices) of X-data 
    Typically, output of function `mblock` from (n, p) data.  
* `bscales` : A vector (of length equal to the nb. 
    of blocks) of the scalars diving the blocks.
* `weights` : Weights (n) of the observations (rows 
    of the blocks). Must be of type `Weight` (see e.g. 
    function `mweight`).

Specificities of each function:
* `fblockscal`: Each block X is tranformed 
    to X / `bscales'.
* `fblockscal_frob`: Let D be the diagonal matrix 
    of vector `weights.w`. Each block X is divided by 
    its Frobenius norm  = sqrt(tr(X' * D * X)). After 
    this scaling, tr(X' * D * X) = 1.
* `fblockscal_mfa`: Each block X is divided by sqrt(lamda),
    where lambda is the dominant eigenvalue of X 
    (this is the "MFA" approach).
* `fblockscal_ncol`: Each block X is divided by the nb. 
    of columns of the block.
* `fblockscal_sd`: Each block X is divided by 
    sqrt(sum(weighted variances of the block-columns)). After 
    this scaling, sum(weighted variances of 
    the block-columns) = 1.

Each function returns the scaled blocks and the scaling values.

## Examples
```julia
using LinearAlgebra

n = 5 ; m = 3 ; p = 10 
X = rand(n, p) 
Xnew = rand(m, p)

listbl = [3:4, 1, [6; 8:10]]
Xbl = mblock(X, listbl) 
Xblnew = mblock(Xnew, listbl) 

Xbl[1]
Xbl[2]
Xbl[3]

bscales = ones(3)
res = fblockscal(Xbl, bscales) ;
res.bscales
res.Xbl[3]
Xbl[3]

w = ones(n)
#w = rand(n)
weights = mweight(w) 

res = fblockscal_frob(Xbl, weights) ;
res.bscales
i = 3
D = Diagonal(weights.w)
tr(res.Xbl[i]' * D * res.Xbl[i])^.5

res = fblockscal_mfa(Xbl, weights) ;
i = 3
res.bscales[i]
pcasvd(Xbl[i], weights).sv[1]

res = fblockscal_ncol(Xbl) ;
res.bscales
i = 3
res.Xbl[i]
Xbl[3] / nco(Xbl[3])

res = fblockscal_sd(Xbl, weights) ;
res.bscales
i = 3
sum(colvar(res.Xbl[i], weights))

## To concatenate the returned blocks
X_concat = reduce(hcat, res.Xbl)
```
"""
function fblockscal(Xbl, bscales)
    Q = eltype(Xbl[1][1, 1])
    nbl = length(Xbl)  
    zXbl = list(Matrix{Q}, nbl)
    @inbounds for k = 1:nbl
        zXbl[k] = copy(ensure_mat(Xbl[k]))
    end
    fblockscal!(zXbl, bscales)
end

function fblockscal!(Xbl::Vector, bscales::Vector)
    nbl = length(Xbl)
    #Threads not faster
    @inbounds for k = 1:nbl
        Xbl[k] ./= bscales[k]
    end
    (Xbl = Xbl, bscales)
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







