struct Knnr
    X::Array{Float64}
    Y::Array{Float64}
    nlvdis::Int
    metric::String
    h::Real
    k::Int
end

"""
    knnr(X, Y; nlvdis, metric, h, k, nlv, verbose = false)
k-Nearest-Neighbours Regression (KNNR).
* `X` : matrix (n, p), or vector (n,).
* `Y` : matrix (n, q), or vector (n,).
* `nlvdis` : The number of LVs to consider in the global PLS used for the dimension reduction before 
    calculating the dissimilarities. If `nlvdis = 0`, there is no dimension reduction.
* `metric` : The type of dissimilarity used for defining the neighbors. Possible values are "eucl" (default; Euclidean distance) 
    and "mahal" (Mahalanobis distance).
* `h` : A scale scalar defining the shape of the weight function. Lower is h, sharper is the function. See function `wdist`.
* `k` : The number of nearest neighbors to select for each observation to predict.
* `verbose` : If true, fitting information are printed.

The predictions are the weighted means over the neighborhood. The weights are defined from 
dissimilarities (e.g. distances) between the new observations to predict and the training observations
in the neighborhood. They are computed fro function 'wdist'.

The dissimilarities used for computing the weights can be 
calculated from the original X-data or after a dimension reduction (argument `nlvdis`). 
In the last case, global PLS scores (LVs) are computed from (X, Y) and the dissimilarities are 
calculated over these scores. For high dimensional X-data, using the Mahalanobis distance often requires 
preliminary dimensionality reduction of the data.

""" 
function knnr(X, Y; nlvdis = 0, metric = "eucl", h = Inf, k = 1)
    return Knnr(X, Y, nlvdis, metric, h, k)
end

function predict(object::Knnr, X)
    X = ensure_mat(X)
    m = size(X, 1)
    q = size(object.Y, 2)
    if(object.nlvdis == 0)
        res = getknn(object.X, X; k = object.k, metric = object.metric)
    else
        fm = plskern(object.X, object.Y; nlv = object.nlvdis)
        res = getknn(fm.T, transform(fm, X); k = object.k, metric = object.metric)
    end
    listw = map(d -> wdist(d, object.h), res.d)
    pred = zeros(m, q)
    @inbounds for i = 1:m
        s = res.ind[i]
        w = listw[i] / sum(listw[i])
        pred[i, :] .= colmeans(@view(object.Y[s, :]), w)
    end
    (pred = pred, listnn = res.ind, listd = res.d, listw = listw)
end

