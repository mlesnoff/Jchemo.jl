"""
    knnr(X, Y; nlvdis = 0, metric = :eucl, h = Inf, k = 1, 
        tol = 1e-4, scal::Bool = false)
k-Nearest-Neighbours regression (KNNR).
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `nlvdis` : Number of latent variables (LVs) to consider in the global PLS 
    used for the dimension reduction before computing the dissimilarities. 
    If `nlvdis = 0`, there is no dimension reduction.
* `metric` : Type of dissimilarity used to select the neighbors and compute
    the weights. Possible values are :eucl (default; Euclidean distance) 
    and :mah (Mahalanobis distance).
* `h` : A scalar defining the shape of the weight function. Lower is h, 
    sharper is the function. See function `wdist`.
* `k` : The number of nearest neighbors to select for each observation to predict.
* `tol` : For stabilization when very close neighbors.
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.
    The scaling is implemented for the global (distances) computations.

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
using JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X 
y = dat.Y.tbc
year = dat.Y.year
tab(year)
s = year .<= 2012
Xtrain = X[s, :]
ytrain = y[s]
Xtest = rmrow(X, s)
ytest = rmrow(y, s)

nlvdis = 20 ; metric = :mah 
h = 2 ; k = 100 ; nlv = 15
fm = knnr(Xtrain, ytrain; nlvdis = nlvdis,
    metric = metric, h = h, k = k) ;
res = Jchemo.predict(fm, Xtest)
rmsep(res.pred, ytest)
f, ax = scatter(vec(pred), ytest)
ablines!(ax, 0, 1)
f
```
""" 
function knnr(X, Y; nlvdis = 0, metric = :eucl, h = Inf, k = 1, 
        scal::Bool = false, tol = 1e-4)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    if nlvdis == 0
        fm = nothing
    else
        fm = plskern(X, Y; nlv = nlvdis, 
            scal = scal)
    end
    return Knnr(X, Y, fm, nlvdis, metric, h, k, tol, scal)
end

"""
    predict(object::Knnr, X)
Compute the Y-predictions from the fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Knnr, X)
    X = ensure_mat(X)
    m = nro(X)
    q = size(object.Y, 2)
    # Getknn
    if isnothing(object.fm)
        if object.scal
            xscales = colstd(object.X)
            zX1 = fscale(object.X, xscales)
            zX2 = fscale(X, xscales)
            res = getknn(zX1, zX2; k = object.k, metric = object.metric)
        else
            res = getknn(object.X, X; k = object.k, metric = object.metric)
        end
    else
        res = getknn(object.fm.T, transf(object.fm, X); k = object.k, 
            metric = object.metric) 
    end
    listw = copy(res.d)
    @inbounds for i = 1:m
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

