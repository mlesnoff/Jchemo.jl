struct LwplsrdaS
    T::Array{Float64}
    y::AbstractMatrix
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
    lwplsrda_s(X, y; reduc = "pls", 
        nlv0, gamma = 1, psamp = 1, samp = "cla", 
        metric = "eucl", h, k, nlv, 
        tol = 1e-4, scal::Bool = false, verbose = false)
kNN-LWPLSR-DA after preliminary (linear or non-linear) dimension 
    reduction (kNN-LWPLSR-DA-S).
* `X` : X-data (n, p).
* `y` : Univariate class membership.
* `reduc` : Type of dimension reduction. Possible values are:
    "pca" (PCA), "pls" (PLS; default), "dkpls" (direct Gaussian kernel PLS).
* `nlv0` : Nb. latent variables (LVs) for preliminary dimension reduction. 
* `gamma` : Scale parameter for the Gaussian kernel when a KPLS is used 
    for dimension reduction. See function `krbf`.
* `psamp` : Proportion of observations sampled in `X, Y`to compute the 
    loadings used to compute the scores.
* `samp` : Type of sampling applied for `psamp`. Possible values are 
    "cla" (stratified random sampling over the classes in `y`; default) 
    or "rand" (random sampling). 
* `metric` : Type of dissimilarity used to select the neighbors and compute
    the weights. Possible values are "eucl" (default; Euclidean distance) 
    and "mahal" (Mahalanobis distance).
* `h` : A scalar defining the shape of the weight function. Lower is h, 
    sharper is the function. See function `wdist`.
* `k` : The number of nearest neighbors to select for each observation to predict.
* `nlv` : Nb. latent variables (LVs) for the models fitted on preliminary 
    scores.
* `tol` : For stabilization when very close neighbors.
* `scal` : Boolean. If `true`, each column of `X` and `Y` 
    is scaled by its uncorrected standard deviation.
    The scaling is implemented for the global (distances) and local (i.e. inside
    each neighborhood) computations.
* `verbose` : If true, fitting information are printed.

This is the same principle as function `lwplsr_s` except that, locally, PLSR-DA models
are fitted instead of PLSR models.

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

fm = lwplsrda_s(Xtrain, ytrain; reduc = "pca", 
    nlv0 = 20, metric = "eucl", h = 2, 
    k = 100, nlv = 10) ;
pred = Jchemo.predict(fm, Xtest).pred
err(pred, ytest)
confusion(pred, ytest).cnt
```
""" 
function lwplsrda_s(X, y; reduc = "pls", 
        nlv0, gamma = 1, psamp = 1, samp = "cla", 
        metric = "eucl", h, k, nlv, 
        tol = 1e-4, scal::Bool = false, verbose = false)
    @assert in(["pca"; "pls"; "dkpls"])(reduc) "Wrong value for argument 'reduc'."    
    @assert psamp >= 0 && psamp <= 1 "psamp must be in [0, 1]"   
    @assert in(["cla"; "rand"])(samp) "Wrong value for argument 'samp'."
    X = ensure_mat(X)
    y = ensure_mat(y)
    n = nro(X)
    s = 1:n
    if psamp < 1
        m = Int64(round(psamp * n))
        if samp == "cla"
            lev = mlev(y)
            nlev = length(lev)
            zm = Int64(round(m / nlev))
            s = sampcla(y, zm).train
        elseif samp == "rand"
            s = sample(1:n, m; replace = false)
        end
    end
    zX = vrow(X, s)
    zy = vrow(y, s)
    if reduc == "pca"
        fm = pcasvd(zX; nlv = nlv0, scal = scal)
    elseif reduc == "pls"
        fm = plsrda(zX, zy; nlv = nlv0, scal = scal)
    elseif reduc == "dkpls"
        fm = dkplsrda(zX, zy; gamma = gamma, nlv = nlv0, 
            scal = scal)
    end
    T = transform(fm, X)
    LwplsrdaS(T, y, fm, metric, h, k, nlv, 
        tol, scal, verbose)
end

"""
    predict(object::LwplsrdaS, X; nlv = nothing)
Compute the Y-predictions from the fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::LwplsrdaS, X; nlv = nothing)
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
    pred = locwlv(object.T, object.y, T; 
        listnn = res.ind, listw = listw, fun = plsrda, nlv = nlv, 
        scal = object.scal, verbose = object.verbose).pred
    (pred = pred, listnn = res.ind, listd = res.d, 
        listw = listw)
end

