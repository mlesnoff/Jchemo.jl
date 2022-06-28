struct LwplsrdaAvg
    X::Array{Float64}
    y::AbstractMatrix
    fm
    metric::String
    h::Real
    k::Int
    nlv::String
    tol::Real
    verbose::Bool
    lev::AbstractVector
    ni::AbstractVector
end

"""
    lwplsrda_avg(X, y; nlvdis, metric, h, k, nlv, 
        tol = 1e-4, verbose = false)
Averaging of kNN-LWPLSR-DA models with different numbers of LVs.
* `X` : X-data.
* `y` : y-data (class membership).
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
* `verbose` : If true, fitting information are printed.

This is the same methodology as for `lwplsr_avg` except that 
PLSR is replaced by PLSR-DA, and the mean is replaced by votes.

For instance, if argument `nlv` is set to `nlv = "5:10"`, the prediction for 
a new observation is the most occurent class within the predictions 
returned by the models with 5 LVS, 6 LVs, ... 10 LVs, respectively.

## Examples
```julia
using JLD2
mypath = dirname(dirname(pathof(Jchemo)))
db = joinpath(mypath, "data", "forages.jld2") 
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

nlvdis = 25 ; metric = "mahal"
h = 2 ; k = 100
nlv = "0:15"
fm = lwplsrda_avg(Xtrain, ytrain;
    nlvdis = nlvdis, metric = metric,
    h = h, k = k, nlv = nlv) ;
pnames(fm)

res = Jchemo.predict(fm, Xtest) ;
pnames(res)
res.pred
err(res.pred, ytest)

res.listnn
res.listd
res.listw
```
""" 
function lwplsrda_avg(X, y; nlvdis, metric, h, k, nlv, 
    tol = 1e-4, verbose = false)
    X = ensure_mat(X)
    y = ensure_mat(y)
    ztab = tab(y)
    if nlvdis == 0
        fm = nothing
    else
        fm = plskern(X, dummy(y).Y; nlv = nlvdis)
    end
    LwplsrdaAvg(X, y, fm, metric, h, k, nlv, tol, verbose,
        ztab.keys, ztab.vals)
end

"""
    predict(object::LwplsrdaAvg, X)
Compute the y-predictions from the fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::LwplsrdaAvg, X) 
    X = ensure_mat(X)
    m = size(X, 1)
    ### Getknn
    if isnothing(object.fm)
        res = getknn(object.X, X; k = object.k, metric = object.metric)
    else
        Tnew = transform(object.fm, X)
        res = getknn(object.fm.T, Tnew; k = object.k, metric = object.metric) 
    end
    listw = copy(res.d)
    for i = 1:m
        w = wdist(res.d[i]; h = object.h)
        w[w .< object.tol] .= object.tol
        listw[i] = w
    end
    ### End
    pred = locw(object.X, object.y, X; 
        listnn = res.ind, listw = listw, fun = plsrda_avg, nlv = object.nlv, 
        verbose = object.verbose).pred
    (pred = pred, listnn = res.ind, listd = res.d, listw = listw)
end







