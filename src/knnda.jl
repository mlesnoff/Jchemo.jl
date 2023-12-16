"""
    knnda(X, y; nlvdis = 0, metric = :eucl, h = Inf, k = 1, tol = 1e-4)
k-Nearest-Neighbours weighted discrimination (kNN-DA).
* `X` : X-data.
* `y` : y-data (class membership).
* `nlvdis` : Number of latent variables (LVs) to consider in the 
    global PLS used for the dimension reduction before 
    calculating the dissimilarities. If `nlvdis = 0`, there is no dimension reduction.
* `metric` : Type of dissimilarity used to select the neighbors. 
    Possible values are :eucl (default; Euclidean distance) 
    and :mah (Mahalanobis distance).
* `h` : A scalar defining the shape of the weight function. Lower is h, 
    sharper is the function. See function `wdist`.
* `k` : The number of nearest neighbors to select for each observation to predict.
* `tol` : For stabilization when very close neighbors.
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.
    The scaling is implemented for the global (distances) computations.

For each new observation to predict:
* i) a number of `k` nearest neighbors (= "weighting 1") is selected
* ii) a weigthed (= "weighting 2") vote is then computed in this neighborhood 
    to select the most frequent class. 

Weightings 1 and 2 are computed from the dissimilarities between the observation 
to predict and the training observations. Depending on argument `nlvdis`, 
the computation is done from the raw X-data or after a dimension reduction. 
In the last case, global PLS2 scores (LVs) are 
computed from {`X`, Y-dummy} (where Y-dummy is the dummy table build from `y`), 
and the dissimilarities are computed over these scores. 

In general, for high dimensional X-data, using the Mahalanobis distance requires 
preliminary dimensionality reduction of the data.

## Examples
```julia
using JLD2
using JchemoData
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/forages2.jld2") 
@load db dat
pnames(dat)

X = dat.X 
Y = dat.Y 
s = Bool.(Y.test)
Xtrain = rmrow(X, s)
ytrain = rmrow(Y.typ, s)
Xtest = X[s, :]
ytest = Y.typ[s]

tab(ytrain)
tab(ytest)

nlvdis = 25 ; metric = :mah
h = 2 ; k = 10
fm = knnda(Xtrain, ytrain;
    nlvdis = nlvdis, metric = metric,
    h = h, k = k) ;
pnames(fm)

res = Jchemo.predict(fm, Xtest) ;
pnames(res)
res.pred
err(res.pred, ytest)
confusion(res.pred, ytest).cnt

res.listnn
res.listd
res.listw
```
""" 
function knnda(X, y; kwargs...) 
    par = recovkwargs(Par, kwargs)
    @assert in([:eucl, :mah])(par.metric) "Wrong value for argument 'metric'."
    X = ensure_mat(X)
    y = ensure_mat(y)
    Q = eltype(X)
    p = nco(X)
    ztab = tab(y)    # only for description
    if par.nlvdis == 0
        fm = nothing
    else
        fm = plskern(X, dummy(y).Y; 
            nlv = par.nlvdis, scal = par.scal)
    end
    xscales = ones(Q, p)
    if par.scal && isnothing(fm)
        xscales .= colstd(X)
    end
    Knnda(X, y, fm, xscales, ztab.keys, 
        ztab.vals, kwargs, par) 
end

"""
    predict(object::Knnda1, X)
Compute the y-predictions from the fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Knnda, X)
    X = ensure_mat(X)
    m = nro(X)
    ## Getknn
    metric = object.par.metric
    h = object.par.h
    k = object.par.k
    tolw = object.par.tolw
    if isnothing(object.fm)
        if object.par.scal
            zX1 = fscale(object.X, object.xscales)
            zX2 = fscale(X, object.xscales)
            res = getknn(zX1, zX2; k, metric)
        else
            res = getknn(object.X, X; k, metric)
        end
    else
        res = getknn(object.fm.T, transf(object.fm, X); 
           k, metric) 
    end
    listw = copy(res.d)
    @inbounds for i = 1:m
        w = wdist(res.d[i]; h)
        w[w .< tolw] .= tolw
        listw[i] = w
    end
    ## End
    pred = similar(object.y, m, 1)
    @inbounds for i = 1:m
        s = res.ind[i]
        pred[i, :] .= findmax_cla(object.y[s], 
            mweight(listw[i]))
    end
    (pred = pred, listnn = res.ind, listd = res.d, 
        listw = listw)
end

