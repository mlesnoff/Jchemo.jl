struct LwmlrS1
    X::Array{Float64}
    Y::Array{Float64}
    fm
    nlv::Int
    gamma::Real
    metric::String
    h::Real
    k::Int
    typ::String
    tol::Real
    scal::Bool
    verbose::Bool
end

"""
    lwmlr_s(X, Y; metric, 
        h, k, nlv, gamma = 1, typ = "pls", 
        tol = 1e-4, scal = false, verbose = false)
kNN-LWMLR after preliminary dimension reduction (kNN-LWMLR-S).
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `metric` : Type of dissimilarity used to select the neighbors and compute
    the weights. Possible values are "eucl" (default; Euclidean distance) 
    and "mahal" (Mahalanobis distance).
* `h` : A scalar defining the shape of the weight function. Lower is h, 
    sharper is the function. See function `wdist`.
* `k` : The number of nearest neighbors to select for each observation to predict.
* `nlv` : Nb. latent variables (LVs) for preliminary dimension reduction. 
* `gamma` :  Scale parameter for the Gaussian kernel when a KPLS is used 
    for dimension reduction. See function `krbf`.
* `typ` : Type of dimension reduction (by default a PLS). Possible values are:
    "pca" (PCA), "pls" (PLS), "dkpls" (Gaussian direct KPLS). 
* `tol` : For stabilization when very close neighbors.
* `verbose` : If true, fitting information are printed.

A kNN-LWMLR is done after preliminary dimension reduction
(parameter `nlv`) of the X-data. When the reduction is done by PCA, 
this corresponds to the "LWR" algorithm proposed by Naes et al. (1990).

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

fm = lwmlr_s(Xtrain, ytrain; metric = "eucl", 
    h = 2, k = 100, nlv = 20, typ = "pca") ;
pred = Jchemo.predict(fm, Xtest).pred
println(rmsep(pred, ytest))
plotxy(vec(pred), ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", 
    ylabel = "Observed (Test)").f  

fm = lwmlr_s(Xtrain, ytrain; metric = "eucl", 
    h = 2, k = 100, nlv = 20, gamma = .01, typ = "dkpls") ;
pred = Jchemo.predict(fm, Xtest).pred
rmsep(pred, ytest)
```
""" 
function lwmlr_s(X, Y; metric, 
        h, k, nlv, gamma = 1, typ = "pls", 
        tol = 1e-4, scal = false, verbose = false)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    if typ == "pca"
        fm = pcasvd(X; nlv = nlv, scal = scal)
    elseif typ == "pls"
        fm = plskern(X, Y; nlv = nlv, scal = scal)
    elseif typ == "dkpls"
        fm = dkplsr(X, Y; gamma = gamma, nlv = nlv, 
            scal = scal)
    end
    LwmlrS1(X, Y, fm, nlv, gamma, metric, h, k, 
        typ, tol, scal, verbose)
end

function predict(object::LwmlrS1, X)
    X = ensure_mat(X)
    m = nro(X)
    if object.typ == "dkpls" 
        Ttrain = object.fm.fm.T
    else
        Ttrain = object.fm.T
    end
    T = transform(object.fm, X)
    # Getknn
    res = getknn(Ttrain, T; 
        k = object.k, metric = object.metric)
    listw = copy(res.d)
    Threads.@threads for i = 1:m
        w = wdist(res.d[i]; h = object.h)
        w[w .< object.tol] .= object.tol
        listw[i] = w
    end
    # End
    pred = locw(Ttrain, object.Y, T; 
        listnn = res.ind, listw = listw, fun = mlr,
        verbose = object.verbose).pred
    (pred = pred, listnn = res.ind, listd = res.d, listw = listw)
end



