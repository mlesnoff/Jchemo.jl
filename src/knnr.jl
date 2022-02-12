struct Knnr
    X::Array{Float64}
    Y::Array{Float64}
    fm
    nlvdis::Int
    metric::String
    h::Real
    k::Int
    tol::Real
end

"""
    knnr(X, Y; nlvdis = 0, metric = "eucl", h = Inf, k = 1, tol = 1e-4)
k-Nearest-Neighbours regression (KNNR).
* `X` : X-data.
* `Y` : Y-data.
* `nlvdis` : Number of latent variables (LVs) to consider in the global PLS used for the dimension reduction before 
    calculating the dissimilarities. If `nlvdis = 0`, there is no dimension reduction.
* `metric` : Type of dissimilarity used to select the neighbors. Possible values are "eucl" (default; Euclidean distance) 
    and "mahal" (Mahalanobis distance).
* `h` : A scalar defining the shape of the weight function. Lower is h, sharper is the function. See function `wdist`.
* `k` : The number of nearest neighbors to select for each observation to predict.
* `tol` : For stabilization when very close neighbors.

For each new observation to predict, the prediction is the weighted mean
over the selected (training) neighborhood. The weights are defined from 
the dissimilarities between the new observation and the selected neighborhood,
and are computed from function 'wdist'.

The dissimilarities can be calculated from the original X-data 
or after a dimension reduction, depending on argument `nlvdis`. 
In the last case, global PLS scores (LVs) are computed from (`X`, `Y`) 
and the dissimilarities are computed over these scores. 
In general, for high dimensional X-data, using the Mahalanobis distance requires 
psuch a reliminary dimensionality reduction.
""" 
function knnr(X, Y; nlvdis = 0, metric = "eucl", h = Inf, k = 1, tol = 1e-4)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    if nlvdis == 0
        fm = nothing
    else
        fm = plskern(X, Y; nlv = nlvdis)
    end
    return Knnr(X, Y, fm, nlvdis, metric, h, k, tol)
end

"""
    predict(object::Knnr, X)
Compute the Y-predictions from the fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Knnr, X)
    X = ensure_mat(X)
    m = size(X, 1)
    q = size(object.Y, 2)
    # Getknn
    if isnothing(object.fm)
        res = getknn(object.X, X; k = object.k, metric = object.metric)
    else
        res = getknn(object.fm.T, transform(object.fm, X); k = object.k, metric = object.metric) 
    end
    listw = copy(res.d)
    for i = 1:m
        w = wdist(res.d[i]; h = object.h)
        w[w .< object.tol] .= object.tol
        listw[i] = w
    end
    # End
    pred = zeros(m, q)
    @inbounds for i = 1:m
        s = res.ind[i]
        w = mweight(listw[i])
        pred[i, :] .= colmean(vrow(object.Y, s), w)
    end
    (pred = pred, listnn = res.ind, listd = res.d, listw = listw)
end

