struct Mbplsr
    fm
    T::Matrix{Float64}
    xmeans::Vector{Vector{Float64}}
    xscales::Vector{Vector{Float64}}
    ymeans::Vector{Float64}
    yscales::Vector{Float64}
    bscales
    weights::Vector{Float64}
end

"""
    mbplsr(X_bl, Y, weights = ones(nro(X_bl[1])); nlv, 
        bscal = "frob", scal = false)
    mbplsr!(X_bl, Y, weights = ones(nro(X_bl[1])); nlv, 
        bscal = "frob", scal = false)
Multiblock PLSR (MBPLSR).
* `X_bl` : List (vector) of blocks (matrices) of X-data. 
    Each component of the list is a block.
* `Y` : Y-data.
* `weights` : Weights of the observations (rows). 
* `nlv` : Nb. latent variables (LVs) to compute.
* `bscal` : Type of block scaling. 
    Possible values are: "none", "frob", "mfa", "ncol", "sd". 
    See functions `blockscal`.
* `scal` : Boolean. If `true`, each column of `X_bl` and `Y` 
    is scaled by its uncorrected standard deviation 
    (before the block scaling).

Usual PLSR on concatened X-blocks, after block scaling.

`weights` is internally normalized to sum to 1. 

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
X_bl = mblock(X, listbl)
# "New" = first two rows of X_bl 
X_bl_new = mblock(X[1:2, :], listbl)

bscal = "frob"
nlv = 5
fm = mbplsr(X_bl, y; nlv = nlv, bscal = bscal) ;
pnames(fm)
fm.T
Jchemo.transform(fm, X_bl_new)
[y Jchemo.predict(fm, X_bl).pred]
Jchemo.predict(fm, X_bl_new).pred
```
"""

function mbplsr(X_bl, Y, weights = ones(nro(X_bl[1])); nlv, 
        bscal = "frob", scal = false)
    nbl = length(X_bl)  
    zX_bl = list(nbl, Matrix{Float64})
    @inbounds for k = 1:nbl
        zX_bl[k] = copy(ensure_mat(X_bl[k]))
    end
    mbplsr!(zX_bl, copy(ensure_mat(Y)), weights; nlv = nlv, 
        bscal = bscal, scal = scal)
end

function mbplsr!(X_bl, Y, weights = ones(nro(X_bl[1])); nlv, 
        bscal = "frob", scal = false)
    nbl = length(X_bl)
    Y = ensure_mat(Y)
    q = nco(Y)
    weights = mweight(weights)
    xmeans = list(nbl, Vector{Float64})
    xscales = list(nbl, Vector{Float64})
    Threads.@threads for k = 1:nbl
        xmeans[k] = colmean(X_bl[k], weights) 
        xscales[k] = ones(nco(X_bl[k]))
        if scal 
            xscales[k] = colstd(X_bl[k], weights)
            X_bl[k] .= cscale(X_bl[k], xmeans[k], xscales[k])
        else
            X_bl[k] .= center(X_bl[k], xmeans[k])
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
        bscal == "frob" ? res = blockscal_frob(X_bl, weights) : nothing
        bscal == "mfa" ? res = blockscal_mfa(X_bl, weights) : nothing
        bscal == "ncol" ? res = blockscal_ncol(X_bl) : nothing
        bscal == "sd" ? res = blockscal_sd(X_bl, weights) : nothing
        X_bl = res.X
        bscales = res.bscales
    end
    X = reduce(hcat, X_bl)
    fm = plskern(X, Y, weights; nlv = nlv, scal = false)
    Mbplsr(fm, fm.T, xmeans, xscales, ymeans, yscales, bscales, weights)
end

""" 
    transform(object::Mbplsr, X_bl; nlv = nothing)
Compute LVs ("scores" T) from a fitted model.
* `object` : The maximal fitted model.
* `X_bl` : A list (vector) of blocks (matrices) of X_bl-data for which LVs are computed.
* `nlv` : Nb. LVs to consider. If nothing, it is the maximum nb. LVs.
""" 
function transform(object::Mbplsr, X_bl; nlv = nothing)
    nbl = length(X_bl)
    zX_bl = list(nbl, Matrix{Float64})
    Threads.@threads for k = 1:nbl
        zX_bl[k] = cscale(X_bl[k], object.xmeans[k], object.xscales[k])
    end
    res = blockscal(zX_bl, object.bscales)
    X = reduce(hcat, res.X)
    transform(object.fm, X; nlv = nlv)
end

"""
    predict(object::Mbplsr, X_bl; nlv = nothing)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X_bl` : A list (vector) of X-data for which predictions are computed.
* `nlv` : Nb. LVs, or collection of nb. LVs, to consider. 
    If nothing, it is the maximum nb. LVs.
""" 
function predict(object::Mbplsr, X_bl; nlv = nothing)
    nbl = length(X_bl)
    zX_bl = list(nbl, Matrix{Float64})
    Threads.@threads for k = 1:nbl
        zX_bl[k] = cscale(X_bl[k], object.xmeans[k], object.xscales[k])
    end
    res = blockscal(zX_bl, object.bscales)
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

