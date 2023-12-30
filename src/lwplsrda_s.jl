"""
    lwplsrda_s(X, y; mreduc = :pls, 
        nlvreduc, gamma = 1, psamp = 1, samp = :cla, 
        metric = :eucl, h, k, nlv, 
        tol = 1e-4, scal::Bool = false, verbose = false)
kNN-LWPLSR-DA after preliminary (linear or non-linear) dimension 
    reduction (kNN-LWPLSR-DA-S).
* `X` : X-data (n, p).
* `y` : Univariate class membership.
* `mreduc` : Type of dimension reduction. Possible values are:
    :pca (PCA), :pls (PLS; default), :dkpls (direct Gaussian kernel PLS).
* `nlvreduc` : Nb. latent variables (LVs) for preliminary dimension reduction. 
* `gamma` : Scale parameter for the Gaussian kernel when a KPLS is used 
    for dimension reduction. See function `krbf`.
* `psamp` : Proportion of observations sampled in `X, Y`to compute the 
    loadings used to compute the scores.
* `samp` : Type of sampling applied for `psamp`. Possible values are 
    :cla (stratified random sampling over the classes in `y`; default) 
    or :rand (random sampling). 
* `metric` : Type of dissimilarity used to select the neighbors and compute
    the weights. Possible values are :eucl (default; Euclidean distance) 
    and :mah (Mahalanobis distance).
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

fm = lwplsrda_s(Xtrain, ytrain; mreduc = :pca, 
    nlvreduc = 20, metric = :eucl, h = 2, 
    k = 100, nlv = 10) ;
pred = Jchemo.predict(fm, Xtest).pred
err(pred, ytest)
confusion(pred, ytest).cnt
```
""" 
function lwplsrda_s(X, y; kwargs...) 
    par = recovkwargs(Par, kwargs) 
    @assert in([:pca; :pls; :dkpls])(par.mreduc) "Wrong value for argument 'mreduc'."    
    @assert 0 <= par.psamp <= 1 "psamp must be in [0, 1]"   
    @assert in([:cla; :rand])(par.msamp) "Wrong value for argument 'msamp'."
    X = ensure_mat(X)
    y = ensure_mat(y)
    n = nro(X)
    ztab = tab(y)    
    s = 1:n
    if par.psamp < 1
        m = Int(round(par.psamp * n))
        if par.msamp == :rand
            s = sample(1:n, m; replace = false)
        elseif par.msamp == :cla
            nlev = length(ztab.keys)
            zm = Int(round(m / nlev))
            s = sampcla(y, zm).test
        end
    end
    zX = vrow(X, s)
    zy = vrow(y, s)
    if par.mreduc == :pca
        fm = pcasvd(zX; nlv = par.nlvreduc, 
            scal = par.scal)
    elseif par.mreduc == :pls
        fm = plskern(zX, dummy(zy).Y; 
            nlv = par.nlvreduc, scal = par.scal)
    elseif par.mreduc == :dkpls
        fm = dkplsr(zX, dummy(zy).Y; 
            kern = :krbf, 
            gamma = par.gamma, nlv = par.nlvreduc, 
            scal = par.scal)
    end
    T = transf(fm, X)
    LwplsrdaS(T, y, fm, ztab.keys, ztab.vals, 
        kwargs, par)
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
    a = object.par.nlv
    isnothing(nlv) ? nlv = a : 
        nlv = (max(minimum(nlv), 0):min(maximum(nlv), a))
    T = transf(object.fm, X)
    ## Getknn
    metric = object.par.metric
    h = object.par.h
    k = object.par.k
    tolw = object.par.tolw
    res = getknn(object.T, T; metric, k)
    listw = copy(res.d)
    Threads.@threads for i = 1:m
        w = wdist(res.d[i]; h, 
            cri = object.par.criw,
            squared = object.par.squared)
        w[w .< tolw] .= tolw
        listw[i] = w
    end
    ## End
    pred = locwlv(object.T, object.y, T; 
        listnn = res.ind, listw = listw, fun = plsrda, 
        nlv = nlv, scal = object.par.scal,
        verbose = object.par.verbose).pred
    (pred = pred, listnn = res.ind, listd = res.d, 
        listw = listw)
end

