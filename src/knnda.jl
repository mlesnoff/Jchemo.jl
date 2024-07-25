"""
    knnda(X, y; kwargs...) 
k-Nearest-Neighbours weighted discrimination (KNN-DA).
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
Keyword arguments:
* `metric` : Type of dissimilarity used to select the 
    neighbors and to compute the weights. Possible values 
    are: `:eucl` (Euclidean distance), `:mah` (Mahalanobis 
    distance).
* `h` : A scalar defining the shape of the weight 
    function computed by function `wdist`. Lower is h, 
    sharper is the function. See function `wdist` for 
    details (keyword arguments `criw` and `squared` of 
    `wdist` can also be specified here).
* `k` : The number of nearest neighbors to select for 
    each observation to predict.
* `tolw` : For stabilization when very close neighbors.
* `scal` : Boolean. If `true`, each column of the global `X` 
    is scaled by its uncorrected standard deviation before 
    the distance and weight computations.

This function has the same principle as function 
`knnr` except that a discrimination is done instead of a 
regression. A weighted vote is done over the neighborhood, 
and the prediction corresponds to the most frequent class.
 
## Examples
```julia
using Jchemo, JchemoData, JLD2
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/forages2.jld2")
@load db dat
pnames(dat)
X = dat.X
Y = dat.Y
n = nro(X) 
s = Bool.(Y.test)
Xtrain = rmrow(X, s)
ytrain = rmrow(Y.typ, s)
Xtest = X[s, :]
ytest = Y.typ[s]
ntrain = nro(Xtrain)
ntest = nro(Xtest)
(ntot = n, ntrain, ntest)
tab(ytrain)
tab(ytest)

metric = :eucl
h = 2 ; k = 10
mod = model(knnda; metric, h, k) 
fit!(mod, Xtrain, ytrain)
pnames(mod)
pnames(mod.fm)
fm = mod.fm ;
fm.lev
fm.ni

res = predict(mod, Xtest) ; 
pnames(res) 
res.listnn
res.listd
res.listw
@head res.pred
@show errp(res.pred, ytest)
conf(res.pred, ytest).cnt

## With dimension reduction
mod1 = model(pcasvd; nlv = 15)
metric = :mah ; h = 1 ; k = 3 
mod2 = model(knnda; metric, h, k) 
mod = pip(mod1, mod2)
fit!(mod, Xtrain, ytrain)
@head pred = predict(mod, Xtest).pred 
errp(pred, ytest)
```
""" 
function knnda(X, y; kwargs...) 
    par = recovkw(ParKnn, kwargs).par
    X = ensure_mat(X)
    y = ensure_mat(y)
    Q = eltype(X)
    p = nco(X)
    taby = tab(y)    
    xscales = ones(Q, p)
    if par.scal && isnothing(fm)
        xscales .= colstd(X)
    end
    Knnda(X, y, xscales, taby.keys, taby.vals, par)
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
        w = wdist(res.d[i]; h, criw, squared)
        w[w .< tolw] .= tolw
        listw[i] = w
        ## New
        #wpr = mweightcla(object.y[res.ind[i]]; prior = object.par.prior).w 
        #listw[i] = wpr
        #listw[i] = sqrt.(w .* wpr)
        ## End
    end
    ## End
    pred = similar(object.y, m, 1)
    @inbounds for i = 1:m
        s = res.ind[i]
        pred[i, :] .= findmax_cla(object.y[s], mweight(listw[i]))
    end
    (pred = pred, listnn = res.ind, listd = res.d, listw)
end

