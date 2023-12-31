"""
    lwmlr_s(; kwargs...)
    lwmlr_s(X, Y; kwargs...)
kNN-LWMLR after preliminary dimension reduction 
    (kNN-LWMLR-S).
* `X` : X-data (n, p).
* `Y` : Y-data (n, q). 
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
* `tolw` : For stabilization when very close neighbors.
* `verbose` : If `true`, fitting information are printed.

The principle is as follows. A preliminary dimension 
reduction (parameter `nlvreduc`) of the X-data (n, p) returns 
a score matrix `T` (n, `nlvreduc`). This dimension reduction can 
be linear (PCA, PLS) or non linear (direct kernel pls DKPLS with
gaussian Kernel), defined in argument `mreduc`. Then, a 
kNN-LWMLR (function `lwmlr`) is done on {`T`, `Y`}.

When n is too large, the reduction dimension can become too 
costly, in particular for a kernel PLS (that requires to compute
a matrix (n, n)). Argument `psamp` allows to sample a proportion 
of the observations on which are computed loadings that are then 
used to compute approximate scores `T` for the all X-data. 

**Note:** The case `mreduc` = `:pca` corresponds to the "LWR" 
algorithm proposed by Naes et al. (1990).

## References 
Naes, T., Isaksson, T., Kowalski, B., 1990. Locally 
weighted regression and scatter correction for near-infrared 
reflectance data. Analytical Chemistry 664–673.

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

mreduc = :pca 
nlvreduc = 20
metric = :eucl 
h = 2 ; k = 100 
mod = lwmlr_s(; mreduc, 
    nlvreduc, metric, h, k) ;
fit!(mod, Xtrain, ytrain)
pnames(mod)
pnames(mod.fm)

res = predict(mod, Xtest) ; 
pnames(res) 
res.listnn
res.listd
res.listw
@head res.pred
@show rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f    

## With non-linear dimension 
## reduction
mreduc = :dkpls
nlvreduc = 20
gamma = .01 
psamp = .5 ; msamp = :rand
metric = :eucl
h = 2 ; k = 100
mod = lwmlr_s(; mreduc, 
    nlvreduc, gamma, psamp, 
    msamp, metric, h, k) ;
fit!(mod, Xtrain, ytrain)
pred = predict(mod, Xtest).pred
@show rmsep(pred, ytest)
plotxy(pred, ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f  
```
""" 
function lwmlr_s(X, Y; kwargs...)
    par = recovkwargs(Par, kwargs)
    @assert in([:pca; :pls; :dkpls])(par.mreduc) "Wrong value for argument 'mreduc'."    
    @assert 0 <= par.psamp <= 1 "psamp must be in [0, 1]"   
    @assert in([:sys; :rand])(par.msamp) "Wrong value for argument 'msamp'."
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n = nro(X)
    s = 1:n
    if par.psamp < 1
        m = Int(round(par.psamp * n))
        if par.msamp == :rand
            s = sample(1:n, m; replace = false)
        elseif par.msamp == :sys
            s = sampsys(rowsum(Y), m).test
        end
    end
    zX = vrow(X, s)
    zY = vrow(Y, s)
    if par.mreduc == :pca
        fm = pcasvd(zX; nlv = par.nlvreduc, 
            scal = par.scal)
    elseif par.mreduc == :pls
        fm = plskern(zX, zY; nlv = par.nlvreduc, 
            scal = par.scal)
    elseif par.mreduc == :dkpls
        fm = dkplsr(zX, zY; kern = :krbf, 
            gamma = par.gamma, nlv = par.nlvreduc, 
            scal = par.scal)
    end
    T = transf(fm, X)
    LwmlrS(T, Y, fm, kwargs, par)
end

"""
    predict(object::LwmlrS, X)
Compute the Y-predictions from the fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::LwmlrS, X)
    X = ensure_mat(X)
    m = nro(X)
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
    pred = locw(object.T, object.Y, T; 
        listnn = res.ind, listw = listw, fun = mlr,
        verbose = object.par.verbose).pred
    (pred = pred, listnn = res.ind, listd = res.d, 
        listw = listw)
end



