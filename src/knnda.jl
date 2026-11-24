"""
    knnda(; kwargs...)
    knnda(X, y::Vector{String}; kwargs...) 
k-Nearest-Neighbours weighted discrimination (kNN-DA).
* `X` : X-data (n, p).
* `y` : Univariate class membership (n). Must be a `Vector{String}`.
Keyword arguments:
* `metric` : Type of dissimilarity used to select the neighbors and to compute the weights (see function `getknn`). 
    Possible values are: `:eucl` (Euclidean), `:mah` (Mahalanobis), `:sam` (spectral angular distance), 
    `:cos` (cosine distance), `:cor` (correlation distance).
* `h` : A scalar defining the shape of the weight function computed by function `winvs`. Lower is h, sharper is the function. 
    See function `winvs` for details (keyword arguments `criw` and `squared` of `winvs` can also be specified here).
* `k` : The number of nearest neighbors to select for each observation to predict.
* `tolw` : For stabilization when very close neighbors.
* `scalx` : Symbol defining the column scaling of the global `X` (before the computation of the distances 
    and the local weights). Possible values are: `:none` (default), `:std` (uncorrected STD), `:prt` (Pareto)
    and `:mad` (MAD).

This function has the same principle as function `knnr` except that a discrimination replaces the regression. A weighted vote 
is done over the neighborhood, and the prediction corresponds to the most frequent class.
 
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

metric = :eucl ; h = 2. ; k = 10
model = knnda(; metric, h, k) 
fit!(model, Xtrain, ytrain)
@names model
@names fitm = model.fitm 
typeof(fitm)
@names fitm

fitm.lev
fitm.ni

res = predict(model, Xtest) ; 
@names res 
res.listnn
res.listd
res.listw
@head res.pred
@show errp(res.pred, ytest)
conf(res.pred, ytest).cnt

# With dimension reduction and function 'pip'
nlv = 15
metric = :cos ; h = 1. ; k = 3 
model1 = pcasvd(; nlv)
model2 = knnda(; metric, h, k) 
model = pip(model1, model2)
fit!(model, Xtrain, ytrain)
@head pred = predict(model, Xtest).pred 
errp(pred, ytest)
```
""" 
knnda(; kwargs...) = JchemoModel(knnda, nothing, kwargs)

function knnda(X, y::Vector{String}; kwargs...) 
    X = ensure_mat(X)
    Q = eltype(X) 
    par = recovkw(ParKnn{Q}, kwargs).par
    taby = tab(y)    
    csx = cscale(X; centrx = :none, scalx = par.scalx)
    Knnda(X, y, csx, taby.n, taby.lev, par)
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
    # Getknn
    metric = object.par.metric
    h = object.par.h
    k = object.par.k
    tolw = object.par.tolw
    criw = object.par.criw
    squared = object.par.squared
    if object.par.scalx != :none
        X1 = fscale(object.X, object.csx.scales)
        X2 = fscale(X, object.csx.scales)
        res = getknn(X1, X2; metric, k)
    else
        res = getknn(object.X, X; metric, k)
    end
    listw = similar(res.d)
    Threads.@threads for i = 1:m
        w = winvs(res.d[i]; h, criw, squared)
        @. w[w < tolw] = tolw
        listw[i] = w
    end
    # End
    pred = similar(object.y, m, 1)
    @inbounds for i = 1:m
        s = res.ind[i]
        pred[i, :] .= findmax_cla(object.y[s], pweight(listw[i]).values)
    end
    (pred = pred, listnn = res.ind, listd = res.d, listw)
end

