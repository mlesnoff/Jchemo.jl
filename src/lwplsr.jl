"""
    lwplsr(X, Y; kwargs...)
k-Nearest-Neighbours locally weighted partial least squares regression 
    (kNN-LWPLSR).
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
Keyword arguments:
* `nlvdis` : Number of latent variables (LVs) to consider 
    in the global PLS used for the dimension reduction 
    before computing the dissimilarities. 
    If `nlvdis = 0`, there is no dimension reduction.
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
* `nlv` : Nb. latent variables (LVs) for the local (i.e. 
    inside each neighborhood) models.
* `scal` : Boolean. If `true`, (a) each column of the global `X` 
    (and of the global `Y` if there is a preliminary PLS reduction dimension) 
    is scaled by its uncorrected standard deviation before to compute 
    the distances and the weights, and (b) the X and Y scaling is also done 
    within each neighborhood (local level) for the weighted PLSR.
* `verbose` : Boolean. If `true`, predicting information
    are printed.

Function `lwplsr` fits kNN-LWPLSR models such as in 
Lesnoff et al. 2020. The general principle of the pipeline 
is as follows (many other variants of pipelines can 
be built):

LWPLSR is a particular case of weighted PLSR (WPLSR) 
(e.g. Schaal et al. 2002). In WPLSR, a priori weights, 
different from the usual 1/n (standard PLSR), are given 
to the n training observations. These weights are used for 
calculating (i) the scores and loadings of the WPLS and 
(ii) the regression model that fits (by weighted 
least squares) the Y-response(s) to the WPLS scores. 
The specificity of LWPLSR (compared to WPLSR) is that the 
weights are computed from dissimilarities (e.g. distances) 
between the new observation to predict and the training 
observations ("L" in LWPLSR comes from "localized"). Note 
that in LWPLSR the weights and therefore the fitted WPLSR 
model change for each new observation to predict.

In the original LWPLSR, all the n training observations 
are used for each observation to predict (e.g. Sicard & Sabatier 2006, 
Kim et al 2011). This can be very time consuming when n is large. 
A faster (and often more efficient) strategy is to preliminary select, 
in the training set, a number of `k` nearest neighbors to the 
observation to predict (= "weighting 1") and then to apply 
LWPLSR only to this pre-selected neighborhood (= "weighting 2"). T
his strategy corresponds to a kNN-LWPLSR and is the one 
implemented in function `lwplsr`.

In `lwplsr`, the dissimilarities used for weightings 1 and 2 
are computed from the raw X-data, or after a dimension reduction,
depending on argument `nlvdis`. In the last case, global PLS2 
scores (LVs) are computed from {`X`, `Y`} and the dissimilarities 
are computed over these scores. 

In general, for high dimensional X-data, using the 
Mahalanobis distance requires preliminary dimensionality 
reduction of the data. In function `knnr', the 
preliminary reduction (argument `nlvdis`) is done by PLS
on {`X`, `Y`}.

## References
Kim, S., Kano, M., Nakagawa, H., Hasebe, S., 2011. Estimation of 
active pharmaceutical ingredients content using locally weighted 
partial least squares and statistical wavelength selection. 
Int. J. Pharm., 421, 269-274.

Lesnoff, M., Metz, M., Roger, J.-M., 2020. Comparison of locally 
weighted PLS strategies for regression and discrimination on 
agronomic NIR data. Journal of Chemometrics, e3209. 
https://doi.org/10.1002/cem.3209

Schaal, S., Atkeson, C., Vijayamakumar, S. 2002. Scalable 
techniques from nonparametric statistics for the real time 
robot learning. Applied Intell., 17, 49-60.

Sicard, E. Sabatier, R., 2006. Theoretical framework for local 
PLS1 regression and application to a rainfall data set. 
Comput. Stat. Data Anal., 51, 1393-1410.

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
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

nlvdis = 5 ; metric = :mah 
h = 1 ; k = 200 ; nlv = 15
model = lwplsr; nlvdis, metric, h, k, nlv) 
fit!(model, Xtrain, ytrain)
pnames(model)
pnames(model.fitm)

res = predict(model, Xtest) ; 
pnames(res) 
res.listnn
res.listd
res.listw
@head res.pred
@show rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5), bisect = true, 
    xlabel = "Prediction", ylabel = "Observed").f    
```
""" 
function lwplsr(X, Y; kwargs...)
    par = recovkw(ParLwplsr, kwargs).par 
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    if par.nlvdis == 0
        fitm = nothing
    else
        fitm = plskern(X, Y; nlv = par.nlvdis, scal = par.scal)
    end
    Q = eltype(X)
    p = nco(X)
    xscales = ones(Q, p)
    if isnothing(fitm) && par.scal
        xscales .= colstd(X)
    end
    Lwplsr(X, Y, fitm, xscales, par)
end

"""
    predict(object::Lwplsr, X; nlv = nothing)
Compute the Y-predictions from the fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Lwplsr, X; nlv = nothing)
    X = ensure_mat(X)
    m = nro(X)
    a = object.par.nlv
    isnothing(nlv) ? nlv = a : nlv = (max(minimum(nlv), 0):min(maximum(nlv), a))
    ## Getknn
    metric = object.par.metric
    h = object.par.h
    k = object.par.k
    tolw = object.par.tolw
    criw = object.par.criw
    squared = object.par.squared
    if isnothing(object.fitm)
        if object.par.scal
            zX1 = fscale(object.X, object.xscales)
            zX2 = fscale(X, object.xscales)
            res = getknn(zX1, zX2; metric, k)
        else
            res = getknn(object.X, X; metric, k)
        end
    else
        res = getknn(object.fitm.T, transf(object.fitm, X); metric, k) 
    end
    listw = copy(res.d)
    #@inbounds for i = 1:m
    Threads.@threads for i = 1:m
        w = wdist(res.d[i]; h, criw, squared)
        w[w .< tolw] .= tolw
        listw[i] = w
    end
    ## End
    pred = locwlv(object.X, object.Y, X; listnn = res.ind, listw, algo = plskern, 
        nlv, scal = object.par.scal, verbose = object.par.verbose).pred
    (pred = pred, listnn = res.ind, listd = res.d, listw)
end

