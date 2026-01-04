"""
    lwplsrda(; kwargs...)
    lwplsrda(X, y; kwargs...)
kNN-LWPLSR-DA.
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
Keyword arguments:
* `nlvdis` : Number of latent variables (LVs) to consider in the global PLS used for the dimension 
    reduction before computing the dissimilarities. If `nlvdis = 0`, there is no dimension reduction.
    If `nlvdis = 0`, there is no dimension reduction.
* `metric` : Type of dissimilarity used to select the neighbors and to compute the weights 
    (see function `getknn`). Possible values are: `:eucl` (Euclidean), `:mah` (Mahalanobis), 
    `:sam` (spectral angular distance), `:cos` (cosine distance), `:cor` (correlation distance).
* `h` : A scalar defining the shape of the weight function computed by function `winvs`. Lower is h, 
    sharper is the function. See function `winvs` for details (keyword arguments `criw` and `squared` of 
    `winvs` can also be specified here).
* `k` : The number of nearest neighbors to select for each observation to predict.
* `tolw` : For stabilization when very close neighbors.
* `nlv` : Nb. latent variables (LVs) for the local (i.e. inside each neighborhood) models.
* `prior` : Type of prior probabilities for class membership. Possible values are: `:unif` (uniform), 
    `:prop` (proportional).
* `scal` : Boolean. If `true`, (a) each column of the global `X` (and of the global `Y` if there 
    is a preliminary PLS reduction dimension) is scaled by its uncorrected standard deviation before to compute 
    the distances and the weights, and (b) the X and Y scaling is also done within each neighborhood (local level) 
    for the weighted PLSR.
* `store` : Boolean. If `true`, the local models fitted on the neighborhoods are stored and returned.
* `verbose` : Boolean. If `true`, predicting information are printed.

This is the same principle as function `lwplsr` except that PLSR-DA models, instead of PLSR models, are fitted 
on the neighborhoods.

## Examples
```julia
using Jchemo, JchemoData, JLD2
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/forages2.jld2")
@load db dat
@names dat
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
h = 2 ; k = 200
nlv = 10
model = lwplsrda(; nlvdis, metric, h, k, nlv, prior = :unif) 
fit!(model, Xtrain, ytrain)
@names model
@names fitm = model.fitm
fitm.lev
fitm.ni
fitm.priors

res = predict(model, Xtest) ; 
@names res 
res.listnn
res.listd
res.listw
@head res.pred
@show errp(res.pred, ytest)
conf(res.pred, ytest).cnt

## Storage of the local models fitted on the neighborhoods
model = lwplsrda(; nlvdis, metric, h, k, nlv, prior = :unif, store = true) 
fit!(model, Xtrain, ytrain)
res = predict(model, Xtest) ; 
@show errp(res.pred, ytest)
@names res
length(res.fitm)
typeof(res.fitm[1]) 
@names res.fitm[1]
@names res.fitm[1].fitm
```
""" 
lwplsrda(; kwargs...) = JchemoModel(lwplsrda, nothing, kwargs)

function lwplsrda(X, y; kwargs...) 
    par = recovkw(ParLwplsda, kwargs).par 
    X = ensure_mat(X)
    y = ensure_mat(y)
    Q = eltype(X)
    taby = tab(y)    
    p = nco(X)
    if par.nlvdis == 0
        fitm = nothing
    else
        weights = mweightcla(vec(y); prior = par.prior)
        fitm = plskern(X, dummy(y).Y, weights; nlv = par.nlvdis, scal = par.scal)
    end
    xscales = ones(Q, p)
    if isnothing(fitm) && par.scal
        xscales .= colstd(X)
    end
    Lwplsrda(fitm, X, y, xscales, taby.keys, taby.vals, par)
end

"""
    predict(object::Lwplsrda, X; nlv = nothing)
Compute the y-predictions from the fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Lwplsrda, X; nlv = nothing)
    X = ensure_mat(X)
    m = nro(X)
    a = object.par.nlv
    isnothing(nlv) ? nlv = a : nlv = min(minimum(nlv), a):min(maximum(nlv), a)
    ## Getknn
    metric = object.par.metric
    h = object.par.h
    k = object.par.k
    tolw = object.par.tolw
    criw = object.par.criw
    squared = object.par.squared
    if isnothing(object.fitm)
        if object.par.scal
            zX1 = fscale(object.X, object.xscales)
            zX2 = fscale(X, object.xscales)
            res = getknn(zX1, zX2; metric, k)
        else
            res = getknn(object.X, X; metric, k)
        end
    else
        res = getknn(object.fitm.T, transf(object.fitm, X); metric, k) 
    end
    listw = copy(res.d)
    Threads.@threads for i = 1:m
    #@inbounds for i = 1:m
        w = winvs(res.d[i]; h, criw, squared)
        w[w .< tolw] .= tolw
        listw[i] = w
    end
    ## End
    ## In each neighborhood, the observation weights in plsrda are given by listw, not by priors
    reslocw = locwlv(object.X, object.y, X; listnn = res.ind, listw, algo = plsrda, nlv, scal = object.par.scal, 
        store = object.par.store, verbose = object.par.verbose)
    (pred = reslocw.pred, fitm = reslocw.fitm, listnn = res.ind, listd = res.d, listw)
end

