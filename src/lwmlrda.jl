struct Lwmlrda1
    X::Array{Float64}
    y::AbstractMatrix
    metric::String
    h::Real
    k::Int
    tol::Real
    verbose::Bool
end

"""
    lwmlrda(X, y; metric = "eucl", h, k, 
        tol = 1e-4, verbose = false)
k-Nearest-Neighbours locally weighted MLR-based discrimination (kNN-LWMLR-DA).
* `X` : X-data (n, p).
* `y` : Univariate class membership.
* `metric` : Type of dissimilarity used to select the neighbors and compute
    the weights. Possible values are "eucl" (default; Euclidean distance) 
    and "mahal" (Mahalanobis distance).
* `h` : A scalar defining the shape of the weight function. Lower is h, 
    sharper is the function. See function `wdist`.
* `k` : The number of nearest neighbors to select for each observation to predict.
* `tol` : For stabilization when very close neighbors.
* `verbose` : If true, fitting information are printed.

This is the same principle as function `lwmlr` except that local MLR-DA models
are fitted instead of local MLR models.

## Examples
```julia
using JLD2

mypath = dirname(dirname(pathof(JchemoData)))
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

nlv = 20
zfm = pcasvd(Xtrain; nlv = nlv) ;
Ttrain = zfm.T 
Ttest = Jchemo.transform(zfm, Xtest)

metric = "mahal"
h = 2 ; k = 100
fm = lwmlrda(Ttrain, ytrain;
    metric = metric, h = h, k = k) ;
res = Jchemo.predict(fm, Ttest) ;
err(res.pred, ytest)

res.listnn
res.listd
res.listw
```
""" 
function lwmlrda(X, y; metric = "eucl", h, k, 
        tol = 1e-4, verbose = false)
    X = ensure_mat(X)
    y = ensure_mat(y)
    Lwmlrda1(X, y, metric, h, k, tol, 
        verbose)
end

"""
    predict(object::Lwmlrda1, X)
Compute y-predictions from the fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Lwmlrda1, X)
    X = ensure_mat(X)
    m = nro(X)
    # Getknn
    res = getknn(object.X, X; 
        k = object.k, metric = object.metric)
    listw = copy(res.d)
    Threads.@threads for i = 1:m
        w = wdist(res.d[i]; h = object.h)
        w[w .< object.tol] .= object.tol
        listw[i] = w
    end
    # End
    pred = locw(object.X, object.y, X; 
        listnn = res.ind, listw = listw, fun = mlrda,
        verbose = object.verbose).pred
    (pred = pred, listnn = res.ind, listd = res.d, 
        listw = listw)
end

