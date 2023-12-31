"""
    lwplsqda(X, y; nlvdis, metric, h, k, nlv, 
        alpha = 0, prior = :unif, tol = 1e-4, scal::Bool = false, 
        verbose = false)
kNN-LWPLS-QDA (with continuum).
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
* `nlvdis` : Number of latent variables (LVs) to consider in the 
    global PLS used for the dimension reduction before 
    calculating the dissimilarities. If `nlvdis = 0`, there is no dimension reduction.
* `metric` : Type of dissimilarity used to select the neighbors. 
    Possible values are :eucl (default; Euclidean distance) 
    and :mah (Mahalanobis distance).
* `h` : A scalar defining the shape of the weight function. Lower is h, 
    sharper is the function. See function `wdist`.
* `k` : The number of nearest neighbors to select for each observation to predict.
* `nlv` : Nb. latent variables (LVs).
* `alpha` : Scalar (∈ [0, 1]) defining the continuum
    between QDA (`alpha = 0`; default) and LDA (`alpha = 1`).
* `prior` : Type of prior probabilities for class membership
    (`unif`: uniform; `prop`: proportional).
* `tol` : For stabilization when very close neighbors.
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.
    The scaling is implemented for the global (distances) and local (i.e. inside
    each neighborhood) computations.
* `verbose` : If `true`, fitting information are printed.

This is the same methodology as for `lwplsr` except that 
PLSR is replaced by PLS-QDA.

The present version of the function suffers from frequent stops
due to non positive definite matrices when doing local QDA. 
The present recommandation is to select a sufficiant large number of neighbors.
This will be fixed in the future.

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
h = 2 ; k = 1000
nlv = 15
fm = lwplsqda(Xtrain, ytrain;
    nlvdis = nlvdis, metric = metric,
    h = h, k = k, nlv = nlv) ;
pnames(fm)

res = Jchemo.predict(fm, Xtest) ;
pnames(res)
res.pred
errp(res.pred, ytest)
confusion(res.pred, ytest).cnt

res.listnn
res.listd
res.listw
```
""" 
function lwplsqda(X, y; kwargs...) 
    par = recovkwargs(Par, kwargs) 
    X = ensure_mat(X)
    y = ensure_mat(y)
    Q = eltype(X)
    ztab = tab(y)    
    p = nco(X)
    if par.nlvdis == 0
        fm = nothing
    else
        fm = plskern(X, dummy(y).Y; 
            nlv = par.nlvdis, scal = par.scal)
    end
    xscales = ones(Q, p)
    if isnothing(fm) && par.scal
        xscales .= colstd(X)
    end
    Lwplsqda(X, y, fm, xscales, ztab.keys, 
        ztab.vals, kwargs, par)
end

"""
    predict(object::Lwplsqda, X; nlv = nothing)
Compute the y-predictions from the fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Lwplsqda, X; nlv = nothing)
    X = ensure_mat(X)
    m = nro(X)
    a = object.par.nlv
    isnothing(nlv) ? nlv = a : 
        nlv = (max(minimum(nlv), 0):min(maximum(nlv), a))
    ## Getknn
    metric = object.par.metric
    h = object.par.h
    k = object.par.k
    tolw = object.par.tolw
    if isnothing(object.fm)
        if object.par.scal
            zX1 = fscale(object.X, object.xscales)
            zX2 = fscale(X, object.xscales)
            res = getknn(zX1, zX2; metric, k)
        else
            res = getknn(object.X, X; metric, k)
        end
    else
        res = getknn(object.fm.T, transf(object.fm, X); 
            metric, k) 
    end
    listw = copy(res.d)
    Threads.@threads for i = 1:m
    #@inbounds for i = 1:m
        w = wdist(res.d[i]; h, 
            cri = object.par.criw,
            squared = object.par.squared)
        w[w .< tolw] .= tolw
        listw[i] = w
    end
    ## End
    pred = locwlv(object.X, object.y, X; 
        listnn = res.ind, listw = listw, fun = plsqda, 
        nlv = nlv, prior = object.par.prior, alpha = object.par.alpha, 
        scal = object.par.scal, verbose = object.par.verbose).pred
    (pred = pred, listnn = res.ind, listd = res.d, 
        listw = listw)
end


