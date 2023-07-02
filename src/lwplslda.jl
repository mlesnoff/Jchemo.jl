struct LwplsLda
    X::Array{Float64}
    y::AbstractMatrix
    fm
    metric::String
    h::Real
    k::Int
    nlv::Int
    prior::String
    tol::Real
    scal::Bool
    verbose::Bool
    lev::AbstractVector
    ni::AbstractVector
end

"""
    lwplslda(X, y; nlvdis, metric, h, k, nlv, prior = "unif", 
        tol = 1e-4, scal = false, verbose = false)
kNN-LWPLS-LDA models.
* `X` : X-data.
* `y` : y-data (class membership).
* `nlvdis` : Number of latent variables (LVs) to consider in the 
    global PLS used for the dimension reduction before 
    calculating the dissimilarities. If `nlvdis = 0`, there is no dimension reduction.
* `metric` : Type of dissimilarity used to select the neighbors. 
    Possible values are "eucl" (default; Euclidean distance) 
    and "mahal" (Mahalanobis distance).
* `h` : A scalar defining the shape of the weight function. Lower is h, 
    sharper is the function. See function `wdist`.
* `k` : The number of nearest neighbors to select for each observation to predict.
* `nlv` : Nb. latent variables (LVs).
* `prior` : Type of prior probabilities for class membership
    (`unif`: uniform; `prop`: proportional).
* `tol` : For stabilization when very close neighbors.
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.
    The scaling is implemented for the global (distances) and local (i.e. inside
    each neighborhood) computations.
* `verbose` : If true, fitting information are printed.

This is the same methodology as for `lwplsr` except that 
PLSR is replaced by PLS-LDA.

## Examples
```julia
using JLD2
using JchemoData
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/forages2.jld2") 
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
nlv = 15
fm = lwplslda(Xtrain, ytrain;
    nlvdis = nlvdis, metric = metric,
    h = h, k = k, nlv = nlv) ;
pnames(fm)

res = Jchemo.predict(fm, Xtest) ;
pnames(res)
res.pred
err(res.pred, ytest)
confusion(res.pred, ytest).cnt

res.listnn
res.listd
res.listw
```
""" 
function lwplslda(X, y; nlvdis, metric, h, k, nlv, prior = "unif", 
        tol = 1e-4, scal = false, verbose = false)
    X = ensure_mat(X)
    y = ensure_mat(y)
    ztab = tab(y)
    if nlvdis == 0
        fm = nothing
    else
        fm = plskern(X, dummy(y).Y; nlv = nlvdis, 
            scal = scal)
    end
    return LwplsLda(X, y, fm, metric, h, k, nlv, prior, tol, 
        scal, verbose, ztab.keys, ztab.vals)
end

"""
    predict(object::LwplsLda, X; nlv = nothing)
Compute the y-predictions from the fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::LwplsLda, X; nlv = nothing)
    X = ensure_mat(X)
    m = nro(X)
    a = object.nlv
    isnothing(nlv) ? nlv = a : nlv = (max(minimum(nlv), 0):min(maximum(nlv), a))
    ## Getknn
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
        res = getknn(object.fm.T, transform(object.fm, X); 
            k = object.k, metric = object.metric) 
    end
    listw = copy(res.d)
    Threads.@threads for i = 1:m
        w = wdist(res.d[i]; h = object.h)
        w[w .< object.tol] .= object.tol
        listw[i] = w
    end
    ## End
    pred = locwlv(object.X, object.y, X; 
        listnn = res.ind, listw = listw, fun = plslda, nlv = nlv, 
        prior = object.prior, scal = object.scal,
        verbose = object.verbose).pred
    (pred = pred, listnn = res.ind, listd = res.d, listw = listw)
end



