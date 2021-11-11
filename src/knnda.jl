struct Knnda2
    X::Array{Float64}
    y::AbstractMatrix
    fm
    nlvdis::Int
    metric::String
    h::Real
    k::Int
    tol::Real
end

"""
    knnda(X, y; nlvdis = 0, metric = "eucl", h = Inf, k = 1, tol = 1e-4)
k-Nearest-Neighbours weighted discrimination (kNN-DA).
* `X` : X-data.
* `y` : y-data (class membership).
* `nlvdis` : Number of latent variables (LVs) to consider in the 
    global PLS used for the dimension reduction before 
    calculating the dissimilarities. If `nlvdis = 0`, there is no dimension reduction.
* `metric` : Type of dissimilarity used to select the neighbors. 
    Possible values are "eucl" (default; Euclidean distance) 
    and "mahal" (Mahalanobis distance).
* `h` : A scalar defining the shape of the weight function. Lower is h, 
    sharper is the function. See function `wdist`.
* `k` : The number of nearest neighbors to select for each observation to predict.
* `tol` : For stabilization when very close neighbors.

For each new observation to predict, i) a number of 
`k` nearest neighbors is selected (= "weighting 1") and ii) a weigthed vote (= "weighting 2") 
is then computed to select the most frequent class in this neighborhood. 

Weightings 1 and 2 are computed from the dissimilarities between the observation 
to predict and the training observations. Depending on argument `nlvdis`, 
the computation is done from the raw X-data or after a dimension reduction. 
In the last case, global PLS2 scores (LVs) are 
computed from {`X`, Y-dummy} (where Y-dummy is the dummy table build from `y`), 
and the dissimilarities are computed over these scores. 

In general, for high dimensional X-data, using the Mahalanobis distance requires 
preliminary dimensionality reduction of the data.
""" 
function knnda(X, y; nlvdis = 0, metric = "eucl", h = Inf, k = 1, tol = 1e-4)
    X = ensure_mat(X)
    y = ensure_mat(y)
    if nlvdis == 0
        fm = nothing
    else
        fm = plskern(X, dummy(y).Y; nlv = nlvdis)
    end
    return Knnda2(X, y, fm, nlvdis, metric, h, k, tol)
end

"""
    predict(object::Knnda1, X)
Compute the y-predictions from the fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Knnda2, X)
    X = ensure_mat(X)
    m = size(X, 1)
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
    pred = similar(object.y, m, 1)
    @inbounds for i = 1:m
        s = res.ind[i]
        pred[i, :] .= findmax_cla(object.y[s], listw[i])
    end
    (pred = pred, listnn = res.ind, listd = res.d, listw = listw)
end

