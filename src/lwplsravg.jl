struct LwplsrAvg
    X::Array{Float64}
    Y::Array{Float64}
    fm
    metric::String
    h::Real
    k::Int
    nlv::String
    typf::String
    typw::String
    alpha::Real
    K::Real
    rep::Real
    tol::Real
    scal::Bool
    verbose::Bool
end

"""
    lwplsravg(X, Y; nlvdis, metric, h, k, nlv, 
        typf = "unif", typw = "bisquare", alpha = 0, K = 5, rep = 10,
        tol = 1e-4, scal = false, verbose = false)
Averaging kNN-LWPLSR models with different numbers of 
    latent variables (LVs).
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `nlvdis` : Number of latent variables (LVs) to consider in the global PLS 
    used for the dimension reduction before calculating the dissimilarities. 
        If `nlvdis = 0`, there is no dimension reduction.
* `metric` : Type of dissimilarity used to select the neighbors. 
    Possible values are "eucl" (default; Euclidean distance) 
    and "mahal" (Mahalanobis distance).
* `h` : A scalar defining the shape of the weight function. Lower is h, 
    sharper is the function. See function `wdist`.
* `k` : The number of nearest neighbors to select for each observation to predict.
* `nlv` : A character string such as "5:20" defining the range of the numbers of LVs 
    to consider ("5:20": the predictions of models with nb LVS = 5, 6, ..., 20 
    are averaged). Syntax such as "10" is also allowed ("10": correponds to 
    the single model with 10 LVs).   
* `tol` : For stabilization when very close neighbors.
* `scal` : Boolean. If `true`, each column of `X` and `Y` 
    is scaled by its uncorrected standard deviation.
    The scaling is implemented for the global (distances) and local (i.e. inside
    each neighborhood) computations.
* `verbose` : If true, fitting information are printed.
*  Other arguments: see ?plsravg.

Ensemblist method where the predictions of each local model are computed 
are computed by averaging or stacking the predictions of a set of models 
built with different numbers of latent variables (LVs).

For instance, if argument `nlv` is set to `nlv = "5:10"`, the prediction for 
a new observation is the average (eventually weighted) or stacking of the predictions 
returned by the models with 5 LVS, 6 LVs, ... 10 LVs, respectively.

See ?plsravg.

## Examples
```julia
using JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
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

nlvdis = 20 ; metric = "mahal" 
h = 1 ; k = 100 ; nlv = "5:15"
fm = lwplsravg(Xtrain, ytrain; nlvdis = nlvdis,
    metric = metric, h = h, k = k, nlv = nlv) ;
res = Jchemo.predict(fm, Xtest)
rmsep(res.pred, ytest)
f, ax = scatter(vec(res.pred), ytest)
ablines!(ax, 0, 1)
f

fm = lwplsravg(Xtrain, ytrain; nlvdis = nlvdis,
    metric = metric, h = h, k = k, nlv = nlv,
    typf = "cv") ;
res = Jchemo.predict(fm, Xtest)
rmsep(res.pred, ytest)
```
""" 
function lwplsravg(X, Y; nlvdis, metric, h, k, nlv, 
    typf = "unif", typw = "bisquare", alpha = 0, K = 5, rep = 10,
    tol = 1e-4, scal = false, verbose = false)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    if nlvdis == 0
        fm = nothing
    else
        fm = plskern(X, Y; nlv = nlvdis, scal = scal)
    end
    LwplsrAvg(X, Y, fm, metric, h, k, nlv, 
        typf, typw, alpha, K, rep, tol, scal, verbose)
end

"""
    predict(object::LwplsrAvg, X)
Compute the Y-predictions from the fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::LwplsrAvg, X) 
    X = ensure_mat(X)
    m = size(X, 1)
    ### Getknn
    if isnothing(object.fm)
        if object.scal
            xscales = colstd(object.X)
            zX1 = scale(object.X, xscales)
            zX2 = scale(X, xscales)
            res = getknn(zX1, zX2; k = object.k, metric = object.metric)
        else
            res = getknn(object.X, X; k = object.k, metric = object.metric)
        end
    else
        res = getknn(object.fm.T, transform(object.fm, X); k = object.k, 
            metric = object.metric) 
    end
    listw = copy(res.d)
    #@inbounds for i = 1:m
    Threads.@threads for i = 1:m
        w = wdist(res.d[i]; h = object.h)
        w[w .< object.tol] .= object.tol
        listw[i] = w
    end
    ### End
    pred = locw(object.X, object.Y, X; 
        listnn = res.ind, listw = listw, fun = plsravg, nlv = object.nlv, 
        typf = object.typf, typw = object.typw,
        alpha = object.alpha, K = object.K, rep = object.rep,
        scal = object.scal,
        verbose = object.verbose).pred
    (pred = pred, listnn = res.ind, listd = res.d, listw = listw)
end

# Not used
function predict_steps(object::LwplsrAvg, X; steps = nothing) 
    X = ensure_mat(X)
    m = size(X, 1)
    ### Getknn
    if isnothing(object.fm)
        res = getknn(object.X, X; k = object.k, metric = object.metric)
    else
        res = getknn(object.fm.T, transform(object.fm, X); k = object.k, metric = object.metric) 
    end
    listw = copy(res.d)
    for i = 1:m
        w = wdist(res.d[i]; h = object.h)
        if isnothing(steps)
            w[w .< object.tol] .= object.tol
        else 
            w[1:steps] .= 1 ; w[(steps + 1):end] .= object.tol
        end
        listw[i] = w
    end
    ### End
    pred = locw(object.X, object.Y, X; 
        listnn = res.ind, listw = listw, nlv = object.nlv, fun = plsravg, 
        typf = object.typf, typw = object.typw, alpha = object.alpha, K = object.K, rep = object.rep,
        verbose = object.verbose).pred
    (pred = pred, listnn = res.ind, listd = res.d, listw = listw)
end






