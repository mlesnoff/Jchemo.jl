"""
    blockscal(Xbl; kwargs...)
    blockscal(Xbl, weights::Weight; kwargs...)
Scale multiblock X-data.
* `Xbl` : List of blocks (vector of matrices) of X-data 
    Typically, output of function `mblock` from (n, p) data.  
* `weights` : Weights (n) of the observations (rows 
    of the blocks). Must be of type `Weight` (see e.g. 
    function `mweight`).
Keyword arguments:
* `bscal` : Type of block scaling. Possible values are:
    `:none`, `:frob`, `:mfa`, `:ncol`, `:sd`. See thereafter.
* `centr` : Boolean. If `true`, each column of blocks in `Xbl` 
    is centered (before the block scaling).
* `scal` : Boolean. If `true`, each column of blocks in `Xbl` 
    is scaled by its uncorrected standard deviation 
    (before the block scaling).

Types of block scaling:
* `:none` : No block scaling. 
* `:frob` : Let D be the diagonal matrix 
    of vector `weights.w`. Each block X is divided by 
    its Frobenius norm  = sqrt(tr(X' * D * X)). After 
    this scaling, tr(X' * D * X) = 1.
* `mfa` : Each block X is divided by sv, where sv is the 
    dominant singular value of X (this is the "MFA" approach).
* `ncol` : Each block X is divided by the nb. 
    of columns of the block.
* `sd` : Each block X is divided by 
    sqrt(sum(weighted variances of the block-columns)). After 
    this scaling, sum(weighted variances of the block-columns) 
    = 1.

## Examples
```julia
using Jchemo
n = 5 ; m = 3 ; p = 10 
X = rand(n, p) 
Xnew = rand(m, p)
listbl = [3:4, 1, [6; 8:10]]
Xbl = mblock(X, listbl) 
Xblnew = mblock(Xnew, listbl) 
@head Xbl[3]

centr = true ; scal = true
bscal = :frob
model = blockscal; centr, scal, bscal)
fit!(model, Xbl)
zXbl = transf(model, Xbl) ; 
@head zXbl[3]

zXblnew = transf(model, Xblnew) ; 
zXblnew[3]
```
"""
function blockscal(Xbl; kwargs...)
    Q = eltype(Xbl[1][1, 1])
    n = nro(Xbl[1])
    weights = mweight(ones(Q, n))
    blockscal(Xbl, weights; kwargs...)
end

function blockscal(Xbl, weights::Weight; kwargs...)
    par = recovkw(ParBlock, kwargs).par
    @assert in([:none, :frob, :mfa, :ncol, :sd])(par.bscal) "Wrong value for argument 'bscal'."
    Q = eltype(Xbl[1][1, 1])
    nbl = length(Xbl)
    xmeans = list(Vector{Q}, nbl)
    xscales = list(Vector{Q}, nbl)
    bscales = ones(Q, nbl)
    bscal = par.bscal
    for k = 1:nbl
        par.centr ? xmeans[k] = colmean(Xbl[k], weights) : xmeans[k] = zeros(Q, nco(Xbl[k]))
        par.scal ? xscales[k] = colstd(Xbl[k], weights) : xscales[k] = ones(Q, nco(Xbl[k]))
        zX = fcscale(Xbl[k], xmeans[k], xscales[k])
        if bscal == :frob
            bscales[k] = frob(zX, weights)
        elseif bscal == :mfa
            sqrtD = Diagonal(sqrt.(weights.w))
            bscales[k] = nipals(sqrtD * zX).sv
        elseif bscal == :ncol
            bscales[k] = nco(zX)
        elseif bscal == :sd
            bscales[k] = sqrt(sum(colvar(zX, weights)))
        end
    end
    Blockscal(bscales, xmeans, xscales, par)
end

""" 
    transf(object::Blockscal, Xbl)
    transf!(object::Blockscal, Xbl)
Compute the preprocessed data from a model.
* `object` : The fitted model.
* `Xbl` : A list of blocks (vector of matrices) 
    of X-data for which LVs are computed.
""" 
function transf(object::Blockscal, Xbl)
    Q = eltype(Xbl[1][1, 1])
    nbl = length(Xbl)  
    zXbl = list(Matrix{Q}, nbl)
    @inbounds for k = 1:nbl
        zXbl[k] = copy(ensure_mat(Xbl[k]))
    end
    transf!(object, zXbl)
    zXbl
end

function transf!(object::Blockscal, Xbl)
    nbl = length(Xbl)  
    @inbounds for k = 1:nbl
        fcscale!(Xbl[k], object.xmeans[k], object.xscales[k])
        Xbl[k] ./= object.bscales[k]
    end
end

"""
    fblockscal(Xbl, bscales)
    fblockscal!(Xbl::Vector, bscales::Vector)
Scale multiblock X-data.
* `Xbl` : List of blocks (vector of matrices) of X-data 
    Typically, output of function `mblock` from (n, p) data.  
* `bscales` : A vector (of length equal to the nb. 
    of blocks) of the scalars diving the blocks.

## Examples
```julia
using Jchemo
n = 5 ; m = 3 ; p = 10 
X = rand(n, p) 
listbl = [3:4, 1, [6; 8:10]]
Xbl = mblock(X, listbl) 

bscales = 10 * ones(3)
zXbl = fblockscal(Xbl, bscales) ;
@head zXbl[3]
@head Xbl[3]

fblockscal!(Xbl, bscales) ;
@head Xbl[3]
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
    Xbl
end 

"""
    mbconcat(Xbl)
Concatenate horizontaly multiblock X-data.
* `Xbl` : List of blocks (vector of matrices) of X-data 
    Typically, output of function `mblock` from (n, p) data.  

## Examples
```julia
using Jchemo
n = 5 ; m = 3 ; p = 10 
X = rand(n, p) 
Xnew = rand(m, p)
listbl = [3:4, 1, [6; 8:10]]
Xbl = mblock(X, listbl) 
Xblnew = mblock(Xnew, listbl) 
@head Xbl[3]

model = mbconcat) 
fit!(model, Xbl)
transf(model, Xbl)
transf(model, Xblnew)
```
"""
function mbconcat(Xbl)
    Mbconcat(nothing)
end

""" 
    transf(object::Mbconcat, Xbl)
Compute the preprocessed data from a model.
* `object` : The fitted model.
* `Xbl` : A list of blocks (vector of matrices) 
    of X-data for which LVs are computed.
""" 
transf(object::Mbconcat, Xbl) = reduce(hcat, Xbl)

"""
    mblock(X, listbl)
Make blocks from a matrix.
* `X` : X-data (n, p).
* `listbl` : A vector whose each component defines 
    the colum numbers defining a block in `X`.
    The length of `listbl` is the number of blocks.

The function returns a list (vector) of blocks.

## Examples
```julia
using Jchemo
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

