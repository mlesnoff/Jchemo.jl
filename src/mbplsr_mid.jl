struct MbplsrMid
    fm
    T::Matrix{Float64}
    fm_bl
    T_bl::Vector{Matrix{Float64}}
    scal::Vector{Float64}
end

"""
    mbplsr_mid(X, Y, weights = ones(size(X[1], 1)); 
        nlvbl, nlv, scaling = false)
Mid-level multiblock PLSR (PLSR on PLS latent variables).
* `X` : List (vector) of blocks (matrices) of X-data. 
    Each component of the list is a block.
* `Y` : Y-data.
* `weights` : Weights of the observations (rows). See `?plskern` for details.
* `nlvbl` : Vector of nb. preliminary latent variables (LVs) to compute in each
    of the blocks. 
* `nlv` : Final nb. LVs to compute.
* `scaling` : Logical indicating if the blocks of the preliminary LVs
    are autoscaled (`blockscal`) or not before te second PLS is done.

Vector `weights` is internally normalized to sum to 1. 
"""
function mbplsr_mid(X, Y, weights = ones(size(X[1], 1)); 
        nlvbl, nlv, scaling = false)
    nbl = length(X)
    fm_bl = list(nbl)
    T_bl = list(nbl, Matrix{Float64})
    @inbounds for i = 1:nbl
        fm_bl[i] = plskern(X[i], Y, weights; nlv = nlvbl[i])
        T_bl[i] = fm_bl[i].T
    end
    if scaling
        res = blockscal(T_bl)
        T_bl = res.X
        scal = res.scal
    else
        scal = ones(nbl)
    end
    zT = reduce(hcat, T_bl)
    fm = plskern(zT, Y, weights; nlv = nlv)
    MbplsrMid(fm, fm.T, fm_bl, T_bl, scal)
end

""" 
    transform(object::MbplsrMid, X; nlv = nothing)
Compute LVs ("scores" T) from a fitted model.
* `object` : The maximal fitted model.
* `X` : A list (vector) of blocks (matrices) of X-data for which LVs are computed.
* `nlv` : Nb. LVs to consider. If nothing, it is the maximum nb. LVs.
""" 
function transform(object::MbplsrMid, X; nlv = nothing)
    a = size(object.T, 2)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    nbl = length(object.fm_bl)
    T_bl = list(nbl, Matrix{Float64})
    @inbounds for i = 1:nbl
        T_bl[i] = transform(object.fm_bl[i], X[i]) / object.scal[i]
    end
    T = reduce(hcat, T_bl)
    transform(object.fm, T)
end

"""
    predict(object::MbplsrMid, X; nlv = nothing)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : A list (vector) of X-data for which predictions are computed.
* `nlv` : Nb. LVs, or collection of nb. LVs, to consider. 
    If nothing, it is the maximum nb. LVs.
""" 
function predict(object::MbplsrMid, X; nlv = nothing)
    a = size(object.T, 2)
    isnothing(nlv) ? nlv = a : nlv = (max(minimum(nlv), 0):min(maximum(nlv), a))
    nbl = length(object.fm_bl)
    T_bl = list(nbl, Matrix{Float64})
    @inbounds for i = 1:nbl
        T_bl[i] = transform(object.fm_bl[i], X[i]) / object.scal[i]
    end
    T = reduce(hcat, T_bl)
    pred = predict(object.fm, T; nlv = nlv).pred
    (pred = pred,)
end



