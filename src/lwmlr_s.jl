struct LwmlrS
    T::Array{Float64}
    Y::Array{Float64}
    fm
    metric::String
    h::Real
    k::Int
    tol::Real
    scal::Bool
    verbose::Bool
end

"""
    lwmlr_s(X, Y; nlv, reduc = "pls", 
        metric = "eucl", h, k, 
        gamma = 1, psamp = 1, samp = "sys", 
        tol = 1e-4, scal = false, verbose = false)
kNN-LWMLR after preliminary (linear or non-linear) dimension 
    reduction (kNN-LWMLR-S).
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
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
* `psamp` : Proportion of observations sampled in `X, Y`to compute the 
    loadings used to compute the scores.
* `samp` : Type of sampling applied for `psamp`. Possible values are: 
    "sys"= systematic grid sampling over `rowsum(Y)`, "random"= random sampling. 
* `tol` : For stabilization when very close neighbors.
* `verbose` : If true, fitting information are printed.

The principle is as follows. A preliminary dimension reduction (parameter `nlv`) 
of the X-data (n, p) returns a score matrix T (n, `nlv`). Then, a kNN-LWMLR 
is done on {T, `Y`}.

The dimension reduction can be linear (PCA, PLS) or non linear (DKPLS), defined 
in argument `reduc`.

When n is too large, the reduction dimension can become too costly,
in particular for a kernel PLS that requires to compute a matrix (n, n).
Argument `psamp` allows to sample a proportion of the observations
that will be used to compute (approximate) scores T for the all X-data. 

The case `reduc = "pca"` corresponds to the "LWR" algorithm proposed 
by Naes et al. (1990).

## References 
Naes, T., Isaksson, T., Kowalski, B., 1990. Locally weighted regression
and scatter correction for near-infrared reflectance data. 
Analytical Chemistry 664–673.

## Examples
```julia
using JchemoData, JLD2, CairoMakie
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X 
y = dat.Y.tbc
year = dat.Y.year
tab(year)
s = year .<= 2012
Xtrain = X[s, :]
ytrain = y[s]
Xtest = rmrow(X, s)
ytest = rmrow(y, s)

fm = lwmlr_s(Xtrain, ytrain; nlv = 20, reduc = "pca", 
    metric = "eucl", h = 2, k = 100) ;
pred = Jchemo.predict(fm, Xtest).pred
println(rmsep(pred, ytest))
plotxy(vec(pred), ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", 
    ylabel = "Observed (Test)").f  

fm = lwmlr_s(Xtrain, ytrain; nlv = 20, reduc = "dkpls", 
    metric = "eucl", h = 2, k = 100, gamma = .01) ;
pred = Jchemo.predict(fm, Xtest).pred
rmsep(pred, ytest)

fm = lwmlr_s(Xtrain, ytrain; nlv = 20, reduc = "dkpls", 
    metric = "eucl", h = 2, k = 100, gamma = .01,
    psamp = .5, samp = "random") ;
pred = Jchemo.predict(fm, Xtest).pred
rmsep(pred, ytest)
```
""" 
function lwmlr_s(X, Y; nlv, reduc = "pls", 
        metric = "eucl", h, k, 
        gamma = 1, psamp = 1, samp = "sys", 
        tol = 1e-4, scal = false, verbose = false)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n = nro(X)
    m = Int64(round(psamp * n))
    if samp == "sys"
        s = sampsys(rowsum(Y); k = m).train
    elseif samp == "random"
        s = sample(1:n, m; replace = false)
    end
    zX = vrow(X, s)
    zY = vrow(Y, s)
    if reduc == "pca"
        fm = pcasvd(zX; nlv = nlv, scal = scal)
    elseif reduc == "pls"
        fm = plskern(zX, zY; nlv = nlv, scal = scal)
    elseif reduc == "dkpls"
        fm = dkplsr(zX, zY; gamma = gamma, nlv = nlv, 
            scal = scal)
    end
    T = transform(fm, X)
    LwmlrS(T, Y, fm, metric, h, k, 
        tol, scal, verbose)
end

function predict(object::LwmlrS, X)
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
    pred = locw(object.T, object.Y, T; 
        listnn = res.ind, listw = listw, fun = mlr,
        verbose = object.verbose).pred
    (pred = pred, listnn = res.ind, listd = res.d, 
        listw = listw)
end



