"""
    lwplsrda_s(; kwargs...) 
    lwplsrda_s(X, y; kwargs...) 
kNN-LWPLSR-DA after preliminary dimension 
    reduction (kNN-LWPLSR-DA-S).
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
Keyword arguments:
* `mreduc` : Type of dimension reduction. Possible 
    values are: `:pca` (PCA), `:pls` (PLS), `:dkpls` 
    (direct Gaussian kernel PLS; see function 
    `dkpls`, and function `krbf` for its keyword 
    argument).
* `nlvreduc` : Nb. latent variables (LVs) for 
    preliminary dimension reduction. 
* `psamp` : Proportion of observations sampled 
    in {`X`, `Y`} to compute the loadings of the 
    preliminary dimension reduction.
* `msamp` : Type of sampling applied when 
    `psamp` < 1. Possible values are: `:sys` 
    (systematic grid sampling over `rowsum(Y)`), 
    `:rand` (random sampling).
* `metric` : Type of dissimilarity used to select the 
    neighbors and to compute the weights. Possible values 
    are: `:eucl` (Euclidean distance), `:mah` (Mahalanobis 
    distance).
* `h` : A scalar defining the shape of the weight 
    function computed by function `wdist`. Lower is h, 
    sharper is the function. See function `wdist` for 
    details (keyword arguments `criw` and `squared` of 
    `wdist` can also be specified here).
* `k` : The number of nearest neighbors to select for 
    each observation to predict.
* `nlv` : Nb. latent variables (LVs) for the local (i.e. 
    inside each neighborhood) models fitted on the 
    preliminary scores.
* `tolw` : For stabilization when very close neighbors.
* `scal` : Boolean. If `true`, each column of `X` 
    and `Y` is scaled by its uncorrected standard deviation
    for the global dimension reduction.

This is the same principle as function `lwplsr_s` except 
that PLSR-DA models, instead of PLSR models, are fitted 
on the neighborhoods.

## Examples
```julia
using JchemoData, JLD2
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/forages2.jld2")
@load db dat
pnames(dat)
X = dat.X
Y = dat.Y
n = nro(X) 
s = Bool.(Y.test)
Xtrain = rmrow(X, s)
ytrain = rmrow(Y.typ, s)
Xtest = X[s, :]
ytest = Y.typ[s]
ntrain = nro(Xtrain)
ntest = nro(Xtest)
(ntot = n, ntrain, ntest)
tab(ytrain)
tab(ytest)

mreduc = :pca ; nlvreduc = 25 ; 
metric = :eucl
h = 2 ; k = 200
nlv = 5
mod = lwplsrda_s(; mreduc, 
    nlvreduc, metric, h, k,
    nlv) 
fit!(mod, Xtrain, ytrain)
pnames(mod)
pnames(mod.fm)
fm = mod.fm ;
fm.lev
fm.ni

res = predict(mod, Xtest) ; 
pnames(res) 
res.listnn
res.listd
res.listw
@head res.pred
errp(res.pred, ytest)
confusion(res.pred, ytest).cnt

## With non-linear dimension 
## reduction
mreduc = :dkpls ; nlvreduc = 25 ;
gamma = 1000 
metric = :eucl
h = 2 ; k = 200
nlv = 5
mod = lwmlrda_s(; mreduc, 
    nlvreduc, gamma, metric, h, k) 
fit!(mod, Xtrain, ytrain)
pred = predict(mod, Xtest).pred
errp(pred, ytest)
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
    taby = tab(y)    
    s = 1:n
    if par.psamp < 1
        m = Int(round(par.psamp * n))
        if par.msamp == :rand
            s = sample(1:n, m; replace = false)
        elseif par.msamp == :cla
            nlev = length(taby.keys)
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
    LwplsrdaS(T, y, fm, taby.keys, taby.vals, 
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
    criw = object.par.criw
    squared = object.par.squared
    res = getknn(object.T, T; metric, k)
    listw = copy(res.d)
    Threads.@threads for i = 1:m
        w = wdist(res.d[i]; h, criw,
            squared)
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

