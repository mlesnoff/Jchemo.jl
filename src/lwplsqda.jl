struct LwplsQda
    X::Array{Float64}
    y::AbstractMatrix
    fm
    metric::String
    h::Real
    k::Int
    nlv::Int
    prior::String
    tol::Real
    verbose::Bool
    lev::AbstractVector
    ni::AbstractVector
end

"""
    lwplsqda(X, y; nlvdis, metric, h, k, nlv, prior = "unif", 
        tol = 1e-4, verbose = false)
kNN-LWPLS-QDA models.
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
* `verbose` : If true, fitting information are printed.

This is the same methodology as for `lwplsr` except that 
PLSR is replaced by PLS-QDA.

The present version of the function suffers from frequent stops
due to non positive definite matrices when doing local QDA. This
will be fixed in the future.  
""" 
function lwplsqda(X, y; nlvdis, metric, h, k, nlv, 
    prior = "unif", tol = 1e-4, verbose = false)
    X = ensure_mat(X)
    y = ensure_mat(y)
    ztab = tab(y)
    if nlvdis == 0
        fm = nothing
    else
        fm = plskern(X, dummy(y).Y; nlv = nlvdis)
    end
    return LwplsQda(X, y, fm, metric, h, k, nlv, prior, tol, verbose,
        ztab.keys, ztab.vals)
end

"""
    predict(object::LwplsQda, X)
Compute the y-predictions from the fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::LwplsQda, X; nlv = nothing)
    X = ensure_mat(X)
    m = size(X, 1)
    a = object.nlv
    isnothing(nlv) ? nlv = a : nlv = (max(minimum(nlv), 0):min(maximum(nlv), a))
    # Getknn
    if isnothing(object.fm)
        res = getknn(object.X, X; k = object.k, metric = object.metric)
    else
        res = getknn(object.fm.T, transform(object.fm, X); 
            k = object.k, metric = object.metric) 
    end
    listw = copy(res.d)
    for i = 1:m
        w = wdist(res.d[i]; h = object.h)
        w[w .< object.tol] .= object.tol
        listw[i] = w
    end
    # End
    pred = locwlv(object.X, object.y, X; 
        listnn = res.ind, listw = listw, fun = plsqda, nlv = nlv, 
        prior = object.prior, verbose = object.verbose).pred
    (pred = pred, listnn = res.ind, listd = res.d, listw = listw)
end



