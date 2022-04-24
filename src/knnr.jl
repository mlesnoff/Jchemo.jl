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
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `nlvdis` : Number of latent variables (LVs) to consider in the global PLS 
    used for the dimension reduction before computing the dissimilarities. 
    If `nlvdis = 0`, there is no dimension reduction.
* `metric` : Type of dissimilarity used to select the neighbors and compute
    the weights. Possible values are "eucl" (default; Euclidean distance) 
    and "mahal" (Mahalanobis distance).
* `h` : A scalar defining the shape of the weight function. Lower is h, 
    sharper is the function. See function `wdist`.
* `k` : The number of nearest neighbors to select for each observation to predict.
* `tol` : For stabilization when very close neighbors.

The function uses functions `getknn` and `locw`; 
see the code for details. Many other variants of kNNR pipelines can be built.
    
The general principle of the method is as follows.

For each new observation to predict, the prediction is the weighted mean
over the selected neighborhood (in `X`). Within the selected neighborhood,
the weights are defined from the dissimilarities between the new observation 
and the neighborhood, and are computed from function 'wdist'.

In general, for high dimensional X-data, using the Mahalanobis distance requires 
preliminary dimensionality reduction of the data.

## Examples
```julia
using JLD2, CairoMakie
mypath = joinpath(@__DIR__, "..", "data")
db = string(mypath, "\\", "cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X 
y = dat.Y.y
year = dat.Y.year
tab(year)
s = year .<= 2012
Xtrain = X[s, :]
ytrain = y[s]
Xtest = rmrow(X, s)
ytest = rmrow(y, s)

nlvdis = 20 ; metric = "mahal" 
h = 2 ; k = 100 ; nlv = 15
fm = knnr(Xtrain, ytrain; nlvdis = nlvdis,
    metric = metric, h = h, k = k) ;
res = predict(fm, Xtest)
rmsep(res.pred, ytest)
f, ax = scatter(vec(pred), ytest)
abline!(ax, 0, 1)
f
```
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

