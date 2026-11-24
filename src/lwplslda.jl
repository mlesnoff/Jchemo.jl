"""
    lwplslda(; kwargs...)
    lwplslda(X, y::Vector{String}; kwargs...)
kNN-LWPLS-LDA.
* `X` : X-data (n, p).
* `y` : Univariate class membership (n). Must be a `Vector{String}`.
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
* `prior` : Type of prior probabilities for class membership. Possible values are: `:unif` (uniform), 
    `:prop` (proportional). This argument only concerns the preliminary global PLS dimension reduction (if any)
    used to compute the distances. In the local models, the priors are not used since the weights are given by the 
    distance-based decreasing weight function.
* `nlv` : Nb. latent variables (LVs) for the local (i.e. inside each neighborhood) models.
* `scalx` : Symbol defining the X column scaling. Possible values are: `:none` (default), `:std` (uncorrected STD), 
    `:prt` (Pareto) and `:mad` (MAD). If not `:none`, (a) each column of the global `X` is scaled (before 
    the computation of the distances and local weights), and (b) the X scaling is also done within each 
    neighborhood (local level) before the weighted PLSR.
* `store` : Boolean. If `true`, the local models fitted on the neighborhoods are stored and returned
    by function `predict`.
* `verbose` : Boolean. If `true`, predicting information are printed.

This is the same principle as function `lwplsr` except that a PLS-LDA model, instead of a PLSR model, is fitted 
on each neighborhoods.

# Examples
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
h = 2. ; k = 200
nlv = 10
model = lwplslda(; nlvdis, metric, h, k, prior = :unif, nlv) 
#model = lwplsqda(; nlvdis, metric, h, k, nlv, alpha = .5) 
fit!(model, Xtrain, ytrain)

res = predict(model, Xtest) ; 
@names res 
res.listnn
res.listd
res.listw
@head res.pred
@show errp(res.pred, ytest)
conf(res.pred, ytest).cnt
```
""" 
lwplslda(; kwargs...) = JchemoModel(lwplslda, nothing, kwargs)

function lwplslda(X, y::Vector{String}; kwargs...) 
    X = ensure_mat(X)
    Q = eltype(X) 
    par = recovkw(ParLwplsda{Q}, kwargs).par 
    taby = tab(y)    
    if par.nlvdis == 0
        priors = nothing
        fitm = nothing
    else
        weights = pweightcla(Q, vec(y); prior = par.prior)
        priors = aggsumv(weights.values, y).val
        fitm = plskern(X, dummy(Q, y).Y, weights; nlv = par.nlvdis, scalx = par.scalx)
    end
    csx = cscale(X; centrx = :none, scalx = par.scalx)
    Lwplslda(fitm, X, y, csx, taby.n, priors, taby.lev, par)
end

"""
    predict(object::Lwplslda, X)
    predict(object::Lwplslda, X, nlv::Union{Int, AbstractVector{Int}})
Compute the y-predictions from the fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Lwplslda, X)
    res = predict(object, X, object.par.nlv)
    (pred = res.pred[1], fitm = res.fitm, listnn = res.listnn, listd = res.listd, 
        listw = res.listw, nlv = res.nlv)
end

function predict(object::Lwplslda, X, nlv::Union{Int, AbstractVector{Int}})
    X = ensure_mat(X)
    m = nro(X)
    a = object.par.nlv
    if isa(nlv, Int)
        nlv = max(1, min(nlv, a))
    else
        nlv = max(1, min(minimum(nlv), a)):min(maximum(nlv), a)
    end
    # Getknn
    metric = object.par.metric
    k = object.par.k
    h = object.par.h
    criw = object.par.criw
    squared = object.par.squared
    tolw = object.par.tolw
    if isnothing(object.fitm)
        if object.par.scalx != :none
            X1 = fscale(object.X, object.csx.scales)
            X2 = fscale(X, object.csx.scales)
            res = getknn(X1, X2; metric, k)
        else
            res = getknn(object.X, X; metric, k)
        end
    else
        res = getknn(object.fitm.T, transf(object.fitm, X); metric, k) 
    end
    listw = similar(res.d)
    Threads.@threads for i = 1:m
        w = winvs(res.d[i]; h, criw, squared)
        @. w[w < tolw] = tolw
        listw[i] = w
    end
    # End
    # In each neighborhood, the observation weights used in 'algo' are given by listw, not by priors
    reslocw = locwlv(object.X, object.y, X; listnn = res.ind, listw, algo = plslda, nlv, 
        scalx = object.par.scalx, store = object.par.store, verbose = object.par.verbose, 
        prior = object.par.prior)
    (pred = reslocw.pred, fitm = reslocw.fitm, listnn = res.ind, listd = res.d, 
        listw, nlv)
end



