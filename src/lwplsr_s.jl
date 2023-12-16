"""
    lwplsr_s(X, Y; reduc = :pls, 
        nlvreduc, gamma = 1, psamp = 1, samp = :sys, 
        metric = :eucl, h, k, nlv, 
        tol = 1e-4, scal::Bool = false, verbose = false)
kNN-LWPLSR after preliminary (linear or non-linear) dimension 
    reduction (kNN-LWPLSR-S).
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `reduc` : Type of preliminary dimension reduction. Possible values are:
    :pca (PCA), :pls (PLS; default), :dkpls (direct Gaussian kernel PLS, see `?dkpls`).
* `nlvreduc` : Nb. latent variables (LVs) for preliminary dimension reduction. 
* `gamma` : Scale parameter for the Gaussian kernel when a KPLS is used 
    for dimension reduction. See function `krbf`.
* `psamp` : Proportion of observations sampled in {`X`, `Y`} to compute the 
    loadings used to compute the scores of the preliminary dimension reduction.
* `samp` : Type of sampling applied for `psamp`. Possible values are: 
    :sys (systematic grid sampling over `rowsum(Y)`) or :rand (random sampling).
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

The principle is as follows. A preliminary dimension reduction (parameter `nlvreduc`) 
of the X-data (n, p) returns a score matrix T (n, `nlv`). Then, a kNN-LWPLSR 
is done on {T, `Y`}. This is a fast approximation of kNN-LWPLSR using the same 
principle as in Shen et al 2019.

The dimension reduction can be linear (PCA, PLS) or non linear (DKPLS), defined 
in argument `reduc`.

When n is too large, the reduction dimension can become too costly,
in particular for a kernel PLS (that requires to compute a matrix (n, n)).
Argument `psamp` allows to sample a proportion of the observations
that will be used to compute (approximate) scores T for the all X-data. 

Setting `nlv = nlvreduc` returns the same predicions as function `lwmlr_s`.

## References
Lesnoff, M., Metz, M., Roger, J.-M., 2020. Comparison of locally weighted PLS 
strategies for regression and discrimination on agronomic NIR data. 
Journal of Chemometrics, e3209. https://doi.org/10.1002/cem.3209

Shen, G., Lesnoff, M., Baeten, V., Dardenne, P., Davrieux, F., Ceballos, H., Belalcazar, J., 
Dufour, D., Yang, Z., Han, L., Pierna, J.A.F., 2019. Local partial least squares based on global PLS scores. 
Journal of Chemometrics 0, e3117. https://doi.org/10.1002/cem.3117

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

nlvreduc = 20 ; metric = :mah 
h = 2 ; k = 100 ; nlv = 10
fm = lwplsr_s(Xtrain, ytrain; nlvreduc = nlvreduc,
    metric = metric, h = h, k = k, nlv = nlv) ;
res = Jchemo.predict(fm, Xtest)
rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", 
    ylabel = "Observed (Test)").f  

fm = lwplsr_s(Xtrain, ytrain; reduc = :dkpls, 
    nlvreduc = nlvreduc, gamma = .1, 
    metric = metric, h = h, k = k, 
    nlv = nlv) ;
res = Jchemo.predict(fm, Xtest)
rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", 
    ylabel = "Observed (Test)").f  

fm = lwplsr_s(Xtrain, ytrain; reduc = :dkpls, 
    nlvreduc = nlvreduc, gamma = .1, psamp = .7, samp = :rand, 
    metric = metric, h = h, k = k,
    nlv = nlv) ;
res = Jchemo.predict(fm, Xtest)
rmsep(res.pred, ytest)
```
""" 
function lwplsr_s(X, Y; kwargs...)
    par = recovkwargs(Par, kwargs)
    @assert in([:pca; :pls; :dkpls])(par.reduc) "Wrong value for argument 'reduc'."    
    @assert 0 <= par.psamp <= 1 "psamp must be in [0, 1]"   
    @assert in([:sys; :rand])(par.msamp) "Wrong value for argument 'samp'." 
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n = nro(X)
    s = 1:n
    if par.psamp < 1
        m = Int(round(par.psamp * n))
        if samp == :rand
            s = sample(1:n, m; replace = false)
        elseif samp == :sys
            s = sampsys(rowsum(Y), m).train
        end
    end
    zX = vrow(X, s)
    zY = vrow(Y, s)
    par.nlv = min(par.nlvreduc, par.nlv)
    if par.reduc == :pca
        fm = pcasvd(zX; nlv = par.nlvreduc, 
            scal = par.scal)
    elseif par.reduc == :pls
        fm = plskern(zX, zY; nlv = par.nlvreduc, 
            scal = par.scal)
    elseif par.reduc == :dkpls
        fm = dkplsr(zX, zY; kern = :krbf, 
            gamma = par.gamma, nlv = par.nlvreduc, 
            scal = par.scal)
    end
    T = transf(fm, X)
    LwplsrS(T, Y, fm, kwargs, par)
end

"""
    predict(object::LwplsrS, X; nlv = nothing)
Compute the Y-predictions from the fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::LwplsrS, X; nlv = nothing)
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
        w = wdist(res.d[i]; h)
        w[w .< tolw] .= tolw
        listw[i] = w
    end
    ## End
    pred = locwlv(object.T, object.Y, T; 
        listnn = res.ind, listw = listw, fun = plskern, 
        nlv = nlv, scal = object.par.scal,
        verbose = object.par.verbose).pred
    (pred = pred, listnn = res.ind, listd = res.d, 
        listw = listw)
end

