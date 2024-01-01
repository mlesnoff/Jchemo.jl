"""
    lwplslda(; kwargs...) 
    lwplslda(X, y; kwargs...)
kNN-LWPLS-LDA.
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
Keyword arguments:
* `nlvdis` : Number of latent variables (LVs) to consider 
    in the global PLS used for the dimension reduction 
    before computing the dissimilarities. 
    If `nlvdis = 0`, there is no dimension reduction.
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
* `nlv` : Nb. latent variables (LVs) for the local (i.e. 
    inside each neighborhood) models.
* `prior` : Type of prior probabilities for class 
    membership. Possible values are: `:unif` (uniform), 
    `:prop` (proportional).
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation
    for the global dimension reduction and the local
    models.

This is the same principle as function `lwplsr` except 
that PLS-LDA models, instead of PLSR models, are fitted 
on the neighborhoods.

## Examples
```julia
using JchemoData, JLD2
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

nlvdis = 25 ; metric = :mah
h = 1 ; k = 100
mod = lwplslda(; nlvdis, 
    metric, h, k, prior = :prop) 
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
errp(res.pred, ytest)
confusion(res.pred, ytest).cnt
```
""" 
function lwplslda(X, y; kwargs...) 
    par = recovkwargs(Par, kwargs) 
    X = ensure_mat(X)
    y = ensure_mat(y)
    Q = eltype(X)
    taby = tab(y)    
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
    Lwplslda(X, y, fm, xscales, taby.keys, 
        taby.vals, kwargs, par)
end

"""
    predict(object::Lwplslda, X; nlv = nothing)
Compute the y-predictions from the fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Lwplslda, X; nlv = nothing)
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
    criw = object.par.criw
    squared = object.par.squared
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
        w = wdist(res.d[i]; h, criw,
            squared)
        w[w .< tolw] .= tolw
        listw[i] = w
    end
    ## End
    pred = locwlv(object.X, object.y, X; 
        listnn = res.ind, listw = listw, fun = plslda, 
        nlv = nlv, prior = object.par.prior, 
        scal = object.par.scal, verbose = object.par.verbose).pred
    (pred = pred, listnn = res.ind, listd = res.d, 
        listw = listw)
end



