struct LwplsrdaS1
    T::Array{Float64}
    Y::Array{Float64}
    fm
    metric::String
    h::Real
    k::Int
    nlv::Int
    tol::Real
    scal::Bool
    verbose::Bool
end

"""
    lwplsrda_s(X, Y; nlv0, reduc = "pls", 
        metric = "eucl", h, k, gamma = 1, psamp = 1, samp = "cla", 
        nlv, tol = 1e-4, scal = false, verbose = false)
kNN-LWPLSR-DA after preliminary (linear or non-linear) dimension 
    reduction (kNN-LWPLSR-DA-S).
* `X` : X-data (n, p).
* `y` : Univariate class membership.
* `nlv0` : Nb. latent variables (LVs) for preliminary dimension reduction. 
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
* `samp` : Type of sampling applied for `psamp`. Possible values are 
    "sys" (systematic grid sampling over `rowsum(Y)`) or "random" (random sampling).
* `nlv` : Nb. latent variables (LVs) for the models fitted on preliminary 
    scores.
* `tol` : For stabilization when very close neighbors.
* `scal` : Boolean. If `true`, each column of `X` and `Y` 
    is scaled by its uncorrected standard deviation.
    The scaling is implemented for the global (distances) and local (i.e. inside
    each neighborhood) computations.
* `verbose` : If true, fitting information are printed.

The principle is as follows. A preliminary dimension reduction (parameter `nlv0`) 
of the X-data (n, p) returns a score matrix T (n, `nlv`). Then, a kNN-LWPLSR 
is done on {T, `Y`}. This is a fast approximation of kNN-LWPLSR using the same 
principle as in Shen et al 2019.

The dimension reduction can be linear (PCA, PLS) or non linear (DKPLS), defined 
in argument `reduc`.

When n is too large, the reduction dimension can become too costly,
in particular for a kernel PLS (that requires to compute a matrix (n, n)).
Argument `psamp` allows to sample a proportion of the observations
that will be used to compute (approximate) scores T for the all X-data. 

Setting `nlv = nlv0` returns the same predicions as function `lwmlr_s`.

## References
Lesnoff, M., Metz, M., Roger, J.-M., 2020. Comparison of locally weighted PLS 
strategies for regression and discrimination on agronomic NIR data. 
Journal of Chemometrics, e3209. https://doi.org/10.1002/cem.3209

Shen, G., Lesnoff, M., Baeten, V., Dardenne, P., Davrieux, F., Ceballos, H., Belalcazar, J., 
Dufour, D., Yang, Z., Han, L., Pierna, J.A.F., 2019. Local partial least squares based on global PLS scores. 
Journal of Chemometrics 0, e3117. https://doi.org/10.1002/cem.3117

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

nlv0 = 20 ; metric = "mahal" 
h = 2 ; k = 100 ; nlv = 10
fm = lwplsrda_s(Xtrain, ytrain; nlv0 = nlv0,
    metric = metric, h = h, k = k, nlv = nlv) ;
res = Jchemo.predict(fm, Xtest)
rmsep(res.pred, ytest)
plotxy(vec(res.pred), ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", 
    ylabel = "Observed (Test)").f  

fm = lwplsrda_s(Xtrain, ytrain; nlv0 = nlv0,
    reduc = "dkpls", metric = metric, 
    h = h, k = k, gamma = .1, nlv = nlv) ;
res = Jchemo.predict(fm, Xtest)
rmsep(res.pred, ytest)
plotxy(vec(res.pred), ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", 
    ylabel = "Observed (Test)").f  

fm = lwplsrda_s(Xtrain, ytrain; nlv0 = nlv0,
    reduc = "dkpls", metric = metric, 
    h = h, k = k, gamma = .1, psamp = .8,
    samp = "random", nlv = nlv) ;
res = Jchemo.predict(fm, Xtest)
rmsep(res.pred, ytest)
```
""" 
function lwplsrda_s(X, Y; nlv0, reduc = "pls", 
        metric = "eucl", h, k, gamma = 1, psamp = 1, samp = "cla", 
        nlv, tol = 1e-4, scal = false, verbose = false)
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
    nlv = min(nlv0, nlv)
    if reduc == "pca"
        fm = pcasvd(zX; nlv = nlv0, scal = scal)
    elseif reduc == "pls"
        fm = plskern(zX, zY; nlv = nlv0, scal = scal)
    elseif reduc == "dkpls"
        fm = dkplsr(zX, zY; gamma = gamma, nlv = nlv0, 
            scal = scal)
    end
    T = transform(fm, X)
    LwplsrdaS1(T, Y, fm, metric, h, k, nlv, 
        tol, scal, verbose)
end

"""
    predict(object::LwplsrdaS1, X; nlv = nothing)
Compute the Y-predictions from the fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::LwplsrdaS1, X; nlv = nothing)
    X = ensure_mat(X)
    m = nro(X)
    a = object.nlv
    isnothing(nlv) ? nlv = a : nlv = (max(0, minimum(nlv)):min(a, maximum(nlv)))
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
    pred = locwlv(object.T, object.Y, T; 
        listnn = res.ind, listw = listw, fun = plskern, nlv = nlv, 
        scal = object.scal, verbose = object.verbose).pred
    (pred = pred, listnn = res.ind, listd = res.d, 
        listw = listw)
end

