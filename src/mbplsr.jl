struct Mbplsr
    fm
    T::Matrix{Float64}
    bscales::Vector{Float64}
    xmeans::Vector{Vector{Float64}}
    xscales::Vector{Vector{Float64}}
    ymeans::Vector{Float64}
    yscales::Vector{Float64}
    weights::Vector{Float64}
end

"""
    mbplsr(Xbl, Y, weights = ones(nro(Xbl[1])); nlv, 
        bscal = "frob", scal = false)
    mbplsr!(Xbl, Y, weights = ones(nro(Xbl[1])); nlv, 
        bscal = "frob", scal = false)
Multiblock PLSR (MBPLSR).
* `Xbl` : List (vector) of blocks (matrices) of X-data. 
    Each component of the list is a block.
* `Y` : Y-data.
* `weights` : Weights of the observations (rows). 
    Internally normalized to sum to 1. 
* `nlv` : Nb. latent variables (LVs) to compute.
* `bscal` : Type of block scaling. 
    Possible values are: "none", "frob", "mfa", "ncol", "sd". 
    See functions `blockscal`.
* `scal` : Boolean. If `true`, each column of blocks in `Xbl` and 
    of `Y` is scaled by its uncorrected standard deviation 
    (before the block scaling).

PLSR (X, Y) where X is the horizontal concatenation of the blocks in `Xbl`.

## Examples
```julia
using JLD2
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "ham.jld2") 
@load db dat
pnames(dat) 

X = dat.X
y = dat.Y.c1
group = dat.group
listbl = [1:11, 12:19, 20:25]
Xbl = mblock(X, listbl)
# "New" = first two rows of Xbl 
Xbl_new = mblock(X[1:2, :], listbl)

bscal = "frob"
nlv = 5
fm = mbplsr(Xbl, y; nlv = nlv, bscal = bscal) ;
pnames(fm)
fm.T
Jchemo.transform(fm, Xbl_new)
[y Jchemo.predict(fm, Xbl).pred]
Jchemo.predict(fm, Xbl_new).pred
```
"""

function mbplsr(Xbl, Y, weights = ones(nro(Xbl[1])); nlv, 
        bscal = "frob", scal = false)
    nbl = length(Xbl)  
    zXbl = list(nbl, Matrix{Float64})
    @inbounds for k = 1:nbl
        zXbl[k] = copy(ensure_mat(Xbl[k]))
    end
    mbplsr!(zXbl, copy(ensure_mat(Y)), weights; nlv = nlv, 
        bscal = bscal, scal = scal)
end

function mbplsr!(Xbl, Y, weights = ones(nro(Xbl[1])); nlv, 
        bscal = "frob", scal = false)
    nbl = length(Xbl)
    Y = ensure_mat(Y)
    q = nco(Y)
    weights = mweight(weights)
    xmeans = list(nbl, Vector{Float64})
    xscales = list(nbl, Vector{Float64})
    Threads.@threads for k = 1:nbl
        xmeans[k] = colmean(Xbl[k], weights) 
        xscales[k] = ones(nco(Xbl[k]))
        if scal 
            xscales[k] = colstd(Xbl[k], weights)
            Xbl[k] .= cscale(Xbl[k], xmeans[k], xscales[k])
        else
            Xbl[k] .= center(Xbl[k], xmeans[k])
        end
    end
    ymeans = colmean(Y, weights)
    yscales = ones(q)
    if scal 
        yscales .= colstd(Y, weights)
        cscale!(Y, ymeans, yscales)
    else
        center!(Y, ymeans)
    end
    if bscal == "none" 
        bscales = ones(nbl)
    else
        bscal == "frob" ? res = blockscal_frob(Xbl, weights) : nothing
        bscal == "mfa" ? res = blockscal_mfa(Xbl, weights) : nothing
        bscal == "ncol" ? res = blockscal_ncol(Xbl) : nothing
        bscal == "sd" ? res = blockscal_sd(Xbl, weights) : nothing
        Xbl = res.X
        bscales = res.bscales
    end
    X = reduce(hcat, Xbl)
    fm = plskern(X, Y, weights; nlv = nlv, scal = false)
    Mbplsr(fm, fm.T, bscales, xmeans, xscales, ymeans, yscales, weights)
end

""" 
    transform(object::Mbplsr, Xbl; nlv = nothing)
Compute latent variables (LVs = scores T) from a fitted model.
* `object` : The fitted model.
* `Xbl` : A list (vector) of blocks (matrices) for which LVs are computed.
* `nlv` : Nb. LVs to consider.
""" 
function transform(object::Mbplsr, Xbl; nlv = nothing)
    nbl = length(Xbl)
    zXbl = list(nbl, Matrix{Float64})
    Threads.@threads for k = 1:nbl
        zXbl[k] = cscale(Xbl[k], object.xmeans[k], object.xscales[k])
    end
    res = blockscal(zXbl, object.bscales)
    X = reduce(hcat, res.X)
    transform(object.fm, X; nlv = nlv)
end

"""
    predict(object::Mbplsr, Xbl; nlv = nothing)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `Xbl` : A list (vector) of X-data for which predictions are computed.
* `nlv` : Nb. LVs, or collection of nb. LVs, to consider. 
""" 
function predict(object::Mbplsr, Xbl; nlv = nothing)
    nbl = length(Xbl)
    zXbl = list(nbl, Matrix{Float64})
    Threads.@threads for k = 1:nbl
        zXbl[k] = cscale(Xbl[k], object.xmeans[k], object.xscales[k])
    end
    res = blockscal(zXbl, object.bscales)
    X = reduce(hcat, res.X)
    W = Diagonal(object.yscales)
    isnothing(nlv) ? le_nlv = 1 : le_nlv = length(nlv) 
    pred = predict(object.fm, X; nlv = nlv).pred
    if le_nlv == 1
        pred = object.ymeans' .+ pred * W 
    else
        # Threads not faster
        @inbounds for i = 1:le_nlv
            pred[i] = object.ymeans' .+ pred[i] * W
        end
    end
    (pred = pred,)
end

