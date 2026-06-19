"""
    mblock(X, listbl)
Make blocks from a matrix.
* `X` : X-data (n, p).
* `listbl` : A vector whose each component defines the colum numbers defining a block in `X`.
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
    @inbounds for k in eachindex(Xbl)
        Xbl[k] = ensure_mat(X[:, listbl[k]])  # 'ensure_mat' is required since the selection can be a vector
    end
    Xbl
end 

"""
    blockscal(; kwargs...)
    blockscal(Xbl; kwargs...)
    blockscal(Xbl::Vector{Matrix{Q}}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: AbstractFloat 
Scale multiblock X-data.
* `Xbl` : List of blocks (vector of matrices) of X-data. Typically, output of function `mblock` from data (n, p).  
* `weights` : Weights (n) of the observations (rows of the blocks). Must be of type `ProbabilityWeights` (see e.g., function `pweight`).
Keyword arguments:
* `centr` : Boolean. If `true`, each column of blocks in `Xbl` is centered (before the block scaling).
* `scal` : Symbol defining the column scaling of `Xbl` (before the block scaling). Possible values are: `:none`, 
    `std` (uncorrected STD), `prt` (pareto) and `:mad` (MAD).
* `bscal` : Type of block scaling. Possible values are: `:none`, `:frob`, `:mfa`, `:ncol`, `:sd`. See thereafter.

If implemented, the data transformations follow the order: column centering, column scaling and finally block scaling. 

Types of block scaling:
* `:none` : No block scaling. 
* `:frob` : Let D be the diagonal matrix of vector `weights.values`. Each block X is divided by its Frobenius norm  = sqrt(tr(X' * D * X)). 
    After this scaling, tr(X' * D * X) = 1.
* `:mfa` : Each block X is divided by sv, where sv is the dominant singular value of X (this is the "MFA" approach; "AFM "in French).
* `:ncol` : Each block X is divided by the nb. of columns of the block.
* `:sd` : Each block X is divided by sqrt(sum(weighted variances of the block-columns)). After this scaling, sum(weighted variances of 
    the block-columns) = 1.

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

centr = true ; scal = :std
bscal = :frob
model = blockscal(; centr, scal, bscal)
fit!(model, Xbl)
## Data transformation
vXbl = transf(model, Xbl) ; 
@head vXbl[3]

zXblnew = transf(model, Xblnew) ; 
zXblnew[3]
```
"""
blockscal(; kwargs...) = JchemoModel(blockscal, nothing, kwargs)

function blockscal(Xbl; kwargs...)
    Xbl = ensure_mat_mb(Xbl)
    n = nro(Xbl[1])
    blockscal(Xbl, pweight(ones(eltype(Xbl[1]), n)); kwargs...)
end

function blockscal(Xbl::Vector{Matrix{Q}}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: AbstractFloat
    par = recovkw(ParBlock, kwargs).par
    @assert in([:none, :frob, :mfa, :ncol, :sd])(par.bscal) "Wrong value for argument 'bscal'."
    nbl = length(Xbl)
    xmeans = list(Vector{Q}, nbl)
    xscales = list(Vector{Q}, nbl)
    bscales = ones(Q, nbl)
    for k in eachindex(Xbl)
        ## Computes:
        ## xmeans for column centering
        zX = copy(Xbl[k])
        pbl = nco(zX)
        xmeans[k] = zeros(Q, pbl)  # if no centering, 'zeros' is required for 'transf'  
        xscales[k] = ones(Q, pbl)
        if par.centr
            xmeans[k] .= colmean(Xbl[k], weights)
            fcenter!(zX, xmeans[k])
        end
        ## xscales for column scaling
        if par.scal != :none
            colscal = def_colscal(par.scal) 
            xscales[k] .= colscal(Xbl[k], weights)
            fscale!(zX, xscales[k])
        end
        ## bscales for block scaling
        if par.bscal == :frob
            bscales[k] = frob(zX, weights)
        elseif par.bscal == :mfa
            fweightr!(zX, sqrt.(weights.values))
            bscales[k] = nipals(zX).sv
        elseif par.bscal == :ncol
            bscales[k] = nco(zX)
        elseif par.bscal == :sd
            bscales[k] = sqrt(sum(colvar(zX, weights)))
        end
    end
    Blockscal(bscales, xmeans, xscales, par)
end

""" 
    transf(object::Blockscal, Xbl)
    transf!(object::Blockscal, Xbl::Vector{Matrix{Q}}) where Q <: AbstractFloat
Compute the preprocessed data from a model.
* `object` : The fitted model.
* `Xbl` : A list of blocks (vector of matrices) of X-data for which LVs are computed.
""" 
function transf(object::Blockscal, Xbl)
    Xbl = ensure_mat_mb(Xbl)
    vXbl = [copy(Xbl[k]) for k in eachindex(Xbl)]
    transf!(object, vXbl)
    vXbl
end

function transf!(object::Blockscal, Xbl::Vector{Matrix{Q}}) where Q <: AbstractFloat 
    @inbounds for k in eachindex(Xbl)
        fcscale!(Xbl[k], object.xmeans[k], object.xscales[k])
        Xbl[k] ./= object.bscales[k]
    end
end

"""
    fblockscal(Xbl, bscales)
    fblockscal!(Xbl::Vector{Matrix{Q}}, bscales::Vector{Q}) where Q <: AbstractFloat
Scale multiblock X-data.
* `Xbl` : List of blocks (vector of matrices) of X-data. Typically, output of function `mblock` from data (n, p).  
* `bscales` : A vector (of length equal to the nb. of blocks) of the scalars diving the blocks.

## Examples
```julia
using Jchemo
n = 5 ; m = 3 ; p = 10 
X = rand(n, p) 
listbl = [3:4, 1, [6; 8:10]]
Xbl = mblock(X, listbl) 

bscales = 10 * ones(3)
vXbl = fblockscal(Xbl, bscales) ;
@head vXbl[3]
@head Xbl[3]

fblockscal!(Xbl, bscales) ;
@head Xbl[3]
```
"""
## This function is not commonly used. See blockscal instead
function fblockscal(Xbl, bscales)
    Xbl = ensure_mat_mb(Xbl)
    nbl = length(Xbl)  
    vXbl = list(Matrix{eltype(Xbl[1])}, nbl)
    @inbounds for k in eachindex(Xbl)
        vXbl[k] = copy(Xbl[k])
    end
    fblockscal!(vXbl, bscales)
end

function fblockscal!(Xbl::Vector{Matrix{Q}}, bscales::Vector{Q}) where Q <: AbstractFloat
    #Threads not faster
    @inbounds for k in eachindex(Xbl)
        Xbl[k] ./= bscales[k]
    end
    Xbl
end 

"""
    mbconcat()
    mbconcat(Xbl)
Concatenate horizontaly multiblock X-data.
* `Xbl` : List of blocks (vector of matrices) of X-data. Typically, output of function `mblock` from data (n, p).  

## Examples
```julia
using Jchemo
n = 5 ; m = 3 ; p = 9 
X = rand(n, p) 
Xnew = rand(m, p)
listbl = [3:4, 1, [6; 8:9]]
Xbl = mblock(X, listbl) 
Xblnew = mblock(Xnew, listbl) 
@head Xbl[3]

model = mbconcat() 
fit!(model, Xbl)
transf(model, Xbl)
transf(model, Xblnew)
```
"""
mbconcat(; kwargs...) = JchemoModel(mbconcat, nothing, kwargs)

function mbconcat(Xbl)
    Mbconcat(nothing)
end

""" 
    transf(object::Mbconcat, Xbl)
Compute the preprocessed data from a model.
* `object` : The fitted model.
* `Xbl` : A list of blocks (vector of matrices) of X-data for which LVs are computed.
""" 
transf(object::Mbconcat, Xbl) = fconcat(Xbl)

"""
    fconcat()
Concatenate horizontaly multiblock X-data.
* `Xbl` : List of blocks (vector of matrices) of X-data. Typically, output of function `mblock` from data (n, p).  

## Examples
```julia
using Jchemo
n = 5 ; m = 3 ; p = 9 
X = rand(n, p) 
Xnew = rand(m, p)
listbl = [3:4, 1, [6; 8:9]]
Xbl = mblock(X, listbl) 
Xblnew = mblock(Xnew, listbl) 
@head Xbl[3]

fconcat(Xbl)
```
"""
fconcat(Xbl) = reduce(hcat, Xbl)

