struct Mbplsr
    fm
    T::Matrix{Float64}
    scal
    xmeans
    weights::Vector{Float64}
end

"""
    mbplsr(X_bl, Y, weights = ones(size(X_bl[1], 1)); nlv, scal = nothing)
Multiblock PLSR (MBPLSR).
* `X_bl` : List (vector) of blocks (matrices) of X-data. 
    Each component of the list is a block.
* `Y` : Y-data.
* `weights` : Weights of the observations (rows). 
* `nlv` : Nb. latent variables (LVs) to compute.
* `bscaling` : Type of block scaling (`"none"`, `"frob"`, `"mfa"`, `"ncol"`, `"sd"`). 
    See functions `blockscal`.

PLSR on scaled and concatened blocks.

`weights` is internally normalized to sum to 1. 
"""
function mbplsr(X_bl, Y, weights = ones(size(X_bl[1], 1)); nlv, bscaling = "none")
    nbl = length(X_bl)
    X = copy(X_bl)
    Y = ensure_mat(Y)
    weights = mweights(weights)
    xmeans = list(nbl, Vector{Float64})
    @inbounds for k = 1:nbl
        xmeans[k] = colmean(X[k], weights)   
        X[k] = center(X[k], xmeans[k])
    end
    if bscaling == "none" 
        scal = ones(nbl)
    else
        bscaling == "frob" ? res = blockscal_frob(X, weights) : nothing
        bscaling == "mfa" ? res = blockscal_frob(X, weights) : nothing
        bscaling == "ncol" ? res = blockscal_sd(X, weights) : nothing
        bscaling == "sd" ? res = blockscal_sd(X, weights) : nothing
        X = res.X
        scal = res.scal
    end
    zX = reduce(hcat, X)
    fm = plskern(zX, Y, weights; nlv = nlv)
    Mbplsr(fm, fm.T, scal, xmeans, weights)
end

""" 
    transform(object::Mbplsr, X_bl; nlv = nothing)
Compute LVs ("scores" T) from a fitted model.
* `object` : The maximal fitted model.
* `X_bl` : A list (vector) of blocks (matrices) of X-data for which LVs are computed.
* `nlv` : Nb. LVs to consider. If nothing, it is the maximum nb. LVs.
""" 
function transform(object::Mbplsr, X_bl; nlv = nothing)
    nbl = length(X_bl)
    X = copy(X_bl)
    @inbounds for k = 1:nbl
        X[k] = center(X[k], object.xmeans[k])
    end
    res = blockscal(X; scal = object.scal)
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
    nbl = length(X_bl)
    X = copy(X_bl)
    @inbounds for k = 1:nbl
        X[k] = center(X[k], object.xmeans[k])
    end
    res = blockscal(X; scal = object.scal)
    zX = reduce(hcat, res.X)
    pred = predict(object.fm, zX; nlv = nlv).pred
    (pred = pred,)
end



