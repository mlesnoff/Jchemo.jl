struct Mbplsr
    fm
    T::Matrix{Float64}
    scal
    weights::Vector{Float64}
end

"""
    mbplsr(X_bl, Y, weights = ones(size(X_bl[1], 1)); nlv, scal = nothing)
Multiblock PLSR.
* `X_bl` : List (vector) of blocks (matrices) of X-data. 
    Each component of the list is a block.
* `Y` : Y-data.
* `weights` : Weights of the observations (rows). 
* `nlv` : Nb. latent variables (LVs) to compute.
* `bscaling` : Type of block scaling (`"none"`, `"frob"`, `"sd"`, `"ncol"`). 
    See functions `blockscal`.

PLSR on scaled and concatened blocks.

Vector `weights` is internally normalized to sum to 1. 
See the help of `plskern` for details.
"""
function mbplsr(X_bl, Y, weights = ones(size(X_bl[1], 1)); nlv, bscaling = "frob")
    Y = ensure_mat(Y)
    weights = mweights(weights)
    nbl = length(X_bl)
    if bscaling == "none"
        scal = ones(nbl)
        X = reduce(hcat, X_bl)
    else
        bscaling == "frob" ? res = blockscal_frob(X_bl, weights) : nothing
        bscaling == "sd" ? res = blockscal_sd(X_bl, weights) : nothing
        bscaling == "ncol" ? res = blockscal_ncol(X_bl) : nothing
        scal = res.scal
        X = reduce(hcat, res.X)    
    end
    fm = plskern(X, Y, weights; nlv = nlv)
    Mbplsr(fm, fm.T, scal, weights)
end

""" 
    transform(object::Mbplsr, X_bl; nlv = nothing)
Compute LVs ("scores" T) from a fitted model.
* `object` : The maximal fitted model.
* `X_bl` : A list (vector) of blocks (matrices) of X-data for which LVs are computed.
* `nlv` : Nb. LVs to consider. If nothing, it is the maximum nb. LVs.
""" 
function transform(object::Mbplsr, X_bl; nlv = nothing)
    res = blockscal(X_bl; scal = object.scal)
    zX = reduce(hcat, res.X)
    transform(object.fm, zX; nlv = nlv)
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
    res = blockscal(X_bl; scal = object.scal)
    zX = reduce(hcat, res.X)
    pred = predict(object.fm, zX; nlv = nlv).pred
    (pred = pred,)
end



