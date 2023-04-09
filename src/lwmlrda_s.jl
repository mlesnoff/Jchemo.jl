struct LwmlrdaS
    T::Array{Float64}
    y::AbstractMatrix
    fm
    metric::String
    h::Real
    k::Int
    tol::Real
    verbose::Bool
end

"""
    lwmlrda_s(X, y; nlv, reduc = "pls", 
        metric = "eucl", h, k, 
        gamma = 1, psamp = 1, samp = "cla", 
        tol = 1e-4, scal = false, verbose = false)
kNN-LWMLR-DA after preliminary (linear or non-linear) dimension 
    reduction (kNN-LWMLR-DA-S).
* `X` : X-data (n, p).
* `y` : Univariate class membership.
* `nlv` : Nb. latent variables (LVs) for preliminary dimension reduction. 
* `reduc` : Type of dimension reduction. Possible values are:
    "pca" (PCA), "pls" (PLS; default), "dkpls" (direct Gaussian kernel PLS).
* `metric` : Type of dissimilarity used to select the neighbors and compute
    the weights. Possible values are "eucl" (default; Euclidean distance) 
    and "mahal" (Mahalanobis distance).
* `h` : A scalar defining the shape of the weight function. Lower is h, 
    sharper is the function. See function `wdist`.
* `k` : The number of nearest neighbors to select for each observation to predict.
* `gamma` : Scale parameter for the Gaussian kernel when a KPLS is used 
    for dimension reduction. See function `krbf`.
* `psamp` : Proportion of observations sampled in `X, y`to compute the 
    loadings used to compute the scores.
* `samp` : Type of sampling applied for `psamp`. Possible values are 
    "cla" (stratified random sampling over the classes in `y`) or "random" (random sampling). 
* `tol` : For stabilization when very close neighbors.
* `verbose` : If true, fitting information are printed.

This is the same principle as function `lwmlr_s` except that, locally, MLR-DA models
are fitted instead of MLR models.

## Examples
```julia
using JLD2

path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/forages.jld2") 
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

fm = lwmlrda_s(Xtrain, ytrain; nlv = 20, reduc = "pca", 
    metric = "eucl", h = 2, k = 100) ;
pred = Jchemo.predict(fm, Xtest).pred
err(pred, ytest)

fm = lwmlrda_s(Xtrain, ytrain; nlv = 20, reduc = "dkpls", 
    metric = "eucl", h = 2, k = 100, gamma = .01) ;
pred = Jchemo.predict(fm, Xtest).pred
err(pred, ytest)

fm = lwmlrda_s(Xtrain, ytrain; nlv = 20, reduc = "dkpls", 
    metric = "eucl", h = 2, k = 100, gamma = .01,
    psamp = .5, samp = "cla") ;
pred = Jchemo.predict(fm, Xtest).pred
err(pred, ytest)
```
""" 
function lwmlrda_s(X, y; nlv, reduc = "pls", 
        metric = "eucl", h, k, 
        gamma = 1, psamp = 1, samp = "cla", 
        tol = 1e-4, scal = false, verbose = false)
    X = ensure_mat(X)
    y = ensure_mat(y)
    n = nro(X)
    m = Int64(round(psamp * n))
    if samp == "cla"
        lev = mlev(y)
        nlev = length(lev)
        zm = Int64(round(m / nlev))
        s = sampcla(y; k = zm).train
    elseif samp == "random"
        s = sample(1:n, m; replace = false)
    end
    zX = vrow(X, s)
    zy = vrow(y, s)
    if reduc == "pca"
        fm = pcasvd(zX; nlv = nlv, scal = scal)
    elseif reduc == "pls"
        fm = plsrda(zX, zy; nlv = nlv, scal = scal)
    elseif reduc == "dkpls"
        fm = dkplsrda(zX, zy; gamma = gamma, nlv = nlv, 
            scal = scal)
    end
    T = transform(fm, X)
    LwmlrdaS(T, y, fm, metric, h, k, 
        tol, verbose)
end

"""
    predict(object::LwmlrdaS, X)
Compute the y-predictions from the fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::LwmlrdaS, X)
    X = ensure_mat(X)
    m = nro(X)
    T = transform(object.fm, X)
    # Getknn
    res = getknn(object.T, T; 
        k = object.k, metric = object.metric)
    listw = copy(res.d)
    Threads.@threads for i = 1:m
        w = wdist(res.d[i]; h = object.h)
        w[w .< object.tol] .= object.tol
        listw[i] = w
    end
    # End
    pred = locw(object.T, object.y, T; 
        listnn = res.ind, listw = listw, fun = mlrda,
        verbose = object.verbose).pred
    (pred = pred, listnn = res.ind, listd = res.d, listw = listw)
end



