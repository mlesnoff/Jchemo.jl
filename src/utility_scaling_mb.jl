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

########### To be removed

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







