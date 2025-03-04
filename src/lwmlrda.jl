"""
    lwmlrda(; kwargs...)
    lwmlrda(X, y; kwargs...) 
k-Nearest-Neighbours locally weighted MLR-based discrimination (kNN-LWMLR-DA).
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
Keyword arguments:
* `metric` : Type of dissimilarity used to select the neighbors and to compute the weights 
    (see function `getknn`). Possible values are: `:eucl` (Euclidean), `:mah` (Mahalanobis), 
    `:sam` (spectral angular distance), `:cor` (correlation distance).
* `h` : A scalar defining the shape of the weight function computed by function `winvs`. Lower is h, 
    sharper is the function. See function `winvs` for details (keyword arguments `criw` and `squared` of 
    `winvs` can also be specified here).
* `k` : The number of nearest neighbors to select for each observation to predict.
* `tolw` : For stabilization when very close neighbors.
* `scal` : Boolean. If `true`, each column of the global `X` 
    is scaled by its uncorrected standard deviation before 
    the distance and weight computations.
* `verbose` : Boolean. If `true`, predicting information are printed.

This is the same principle as function `lwmlr` except 
that MLR-DA models, instead of MLR models, are fitted 
on the neighborhoods.

## Examples
```julia
using Jchemo, JchemoData, JLD2
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/iris.jld2")
@load db dat
@names dat
@head dat.X
X = dat.X[:, 1:4]
y = dat.X[:, 5]
n = nro(X)
ntest = 30
s = samprand(n, ntest)
Xtrain = X[s.train, :]
ytrain = y[s.train]
Xtest = X[s.test, :]
ytest = y[s.test]
ntrain = n - ntest
(ntot = n, ntrain, ntest)
tab(ytrain)
tab(ytest)

metric = :mah
h = 2 ; k = 10
model = lwmlrda(; metric, h, k) 
fit!(model, Xtrain, ytrain)
@names model
@names model.fitm
fitm = model.fitm ;
fitm.lev
fitm.ni

res = predict(model, Xtest) ; 
@names res 
res.listnn
res.listd
res.listw
@head res.pred
errp(res.pred, ytest)
conf(res.pred, ytest).cnt
```
""" 
lwmlrda(; kwargs...) = JchemoModel(lwmlrda, nothing, kwargs)

function lwmlrda(X, y; kwargs...) 
    par = recovkw(ParKnn, kwargs).par
    X = ensure_mat(X)
    y = ensure_mat(y)
    taby = tab(y)
    Q = eltype(X)
    p = nco(X)
    xscales = ones(Q, p)
    if par.scal
        xscales .= colstd(X)
    end
    Lwmlrda(X, y, xscales, taby.keys, taby.vals, par)  
end

"""
    predict(object::Lwmlrda, X)
Compute y-predictions from the fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Lwmlrda, X)
    X = ensure_mat(X)
    m = nro(X)
    ## Getknn
    metric = object.par.metric
    h = object.par.h
    k = object.par.k
    tolw = object.par.tolw
    criw = object.par.criw
    squared = object.par.squared
    if object.par.scal
        zX1 = fscale(object.X, object.xscales)
        zX2 = fscale(X, object.xscales)
        res = getknn(zX1, zX2; metric, k)
    else
        res = getknn(object.X, X; metric, k)
    end
    listw = copy(res.d)
    Threads.@threads for i = 1:m
        w = winvs(res.d[i]; h, criw, squared)
        w[w .< tolw] .= tolw
        listw[i] = w
    end
    ## End
    pred = locw(object.X, object.y, X; listnn = res.ind, listw, algo = mlrda, 
        verbose = object.par.verbose).pred
    (pred = pred, listnn = res.ind, listd = res.d, listw)
end

