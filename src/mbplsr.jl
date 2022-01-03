struct Mbplsr
    fm
    T::Matrix{Float64}
    scal
    weights::Vector{Float64}
end

"""
    mbplsr(X, Y, weights = ones(size(X[1], 1)); nlv, scal = nothing)
Multiblock PLSR (scaling and concatenation of blocks).
* `X` : List (vector) of blocks (matrices) of X-data. 
    Each component of the list is a block.
* `Y` : Y-data.
* `weights` : Weights of the observations (rows). 
* `nlv` : Nb. latent variables (LVs) to compute.
* `scal` : If `nothing`, each block is "autoscaled" (i.e. divided) by the 
    the square root of the sum of the variances of each column of the block.
    Else, `scal` must be a vector defining the scaling values dividing the blocks.

Vector `weights` (row-weighting) is internally normalized to sum to 1. 
See the help of `plskern` for details.
"""
function mbplsr(X, Y, weights = ones(size(X[1], 1)); nlv, scal = nothing)
    Y = ensure_mat(Y)
    weights = mweights(weights)
    res = blockscal(X, weights; scal = scal)
    zX = reduce(hcat, res.X)
    fm = plskern(zX, Y, weights; nlv = nlv)
    Mbplsr(fm, fm.T, res.scal, weights)
end

""" 
    transform(object::Mbplsr, X; nlv = nothing)
Compute LVs ("scores" T) from a fitted model.
* `object` : The maximal fitted model.
* `X` : A list (vector) of blocks (matrices) of X-data for which LVs are computed.
* `nlv` : Nb. LVs to consider. If nothing, it is the maximum nb. LVs.
""" 
function transform(object::Mbplsr, X; nlv = nothing)
    res = blockscal(X; scal = object.scal)
    zX = reduce(hcat, res.X)
    transform(object.fm, zX; nlv = nlv)
end

"""
    predict(object::Mbplsr, X; nlv = nothing)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : A list (vector) of X-data for which predictions are computed.
* `nlv` : Nb. LVs, or collection of nb. LVs, to consider. 
    If nothing, it is the maximum nb. LVs.
""" 
function predict(object::Mbplsr, X; nlv = nothing)
    res = blockscal(X; scal = object.scal)
    zX = reduce(hcat, res.X)
    pred = predict(object.fm, zX; nlv = nlv).pred
    (pred = pred,)
end



