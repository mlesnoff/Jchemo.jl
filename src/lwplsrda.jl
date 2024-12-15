"""
    lwplsrda(; kwargs...)
    lwplsrda(X, y; kwargs...)
kNN-LWPLSR-DA.
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
    function computed by function `wdis`. Lower is h, 
    sharper is the function. See function `wdis` for 
    details (keyword arguments `criw` and `squared` of 
    `wdis` can also be specified here).
* `k` : The number of nearest neighbors to select for 
    each observation to predict.
* `tolw` : For stabilization when very close neighbors.
* `nlv` : Nb. latent variables (LVs) for the local (i.e. 
    inside each neighborhood) models.
* `prior` : Type of prior probabilities for class 
    membership. Possible values are: `:unif` (uniform), 
    `:prop` (proportional).
* `scal` : Boolean. If `true`, (a) each column of the global `X` 
    (and of the global Ydummy if there is a preliminary PLS reduction dimension) 
    is scaled by its uncorrected standard deviation before to compute 
    the distances and the weights, and (b) the X and Ydummy scaling is also done 
    within each neighborhood (local level) for the weighted PLS.
* `verbose` : Boolean. If `true`, predicting information
    are printed.

This is the same principle as function `lwplsr` except 
that PLSR-DA models, instead of PLSR models, are fitted 
on the neighborhoods.

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

nlvdis = 25 ; metric = :mah
h = 2 ; k = 200
nlv = 10
model = lwplsrda(; nlvdis, metric, h, k, nlv) 
fit!(model, Xtrain, ytrain)
pnames(model)
pnames(model.fitm)
fitm = model.fitm ;
fitm.lev
fitm.ni

res = predict(model, Xtest) ; 
pnames(res) 
res.listnn
res.listd
res.listw
@head res.pred
@show errp(res.pred, ytest)
conf(res.pred, ytest).cnt
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
    Lwplsrda(X, y, fitm, xscales, taby.keys, taby.vals, par)
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
    isnothing(nlv) ? nlv = a : nlv = (max(minimum(nlv), 0):min(maximum(nlv), a))
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
        w = wdis(res.d[i]; h, criw, squared)
        w[w .< tolw] .= tolw
        listw[i] = w
    end
    ## End
    ## In each neighborhood, the observation weights in plsrda are given by listw, not by priors
    pred = locwlv(object.X, object.y, X; listnn = res.ind, listw, algo = plsrda, 
        nlv, scal = object.par.scal, verbose = object.par.verbose).pred
    (pred = pred, listnn = res.ind, listd = res.d, listw)
end

