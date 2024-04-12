"""
    knnr(X, Y; kwargs...) 
k-Nearest-Neighbours weighted regression (KNNR).
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
* `scal` : Boolean. If `true`, each column of `X` 
    and `Y` is scaled by its uncorrected standard deviation
    for the global dimension reduction.

The general principle of this function is as 
follows (many other variants of kNNR pipelines 
can be built):

For each new observation to predict, the prediction is the 
weighted mean over a selected neighborhood (in `X`) of 
size `k`. Within the selected neighborhood, the weights 
are defined from the dissimilarities between the new 
observation and the neighborhood, and are computed from 
function 'wdist'.
    
In general, for high dimensional X-data, using the 
Mahalanobis distance requires preliminary dimensionality 
reduction of the data. In function `knnr', the 
preliminary reduction (argument `nlvdis`) is done by PLS
on {`X`, `Y`}.

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

nlvdis = 5 ; metric = :mah 
#nlvdis = 0 ; metric = :eucl 
h = 1 ; k = 5 
mod = model(knnr; nlvdis, metric, h, k) ;
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
plotxy(res.pred, ytest; color = (:red, .5), bisect = true, 
    xlabel = "Prediction", ylabel = "Observed").f    

####### Example of fitting the function sinc(x)
####### described in Rosipal & Trejo 2001 p. 105-106 
x = collect(-10:.2:10) 
x[x .== 0] .= 1e-5
n = length(x)
zy = sin.(abs.(x)) ./ abs.(x) 
y = zy + .2 * randn(n) 
mod = model(knnr; k = 15, h = 5) 
fit!(mod, x, y)
pred = predict(mod, x).pred 
f, ax = scatter(x, y) 
lines!(ax, x, zy, label = "True model")
lines!(ax, x, vec(pred), label = "Fitted model")
axislegend("Method")
f
```
""" 
function knnr(X, Y; kwargs...) 
    par = recovkwargs(Par, kwargs)
    @assert in([:eucl, :mah])(par.metric) "Wrong value for argument 'metric'."
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    Q = eltype(X)
    p = nco(X)
    if par.nlvdis == 0
        fm = nothing
    else
        fm = plskern(X, Y; nlv = par.nlvdis, scal = par.scal)
    end
    xscales = ones(Q, p)
    if par.scal && isnothing(fm)
        xscales .= colstd(X)
    end
    Knnr(X, Y, fm, xscales, kwargs, par)
end

"""
    predict(object::Knnr, X)
Compute the Y-predictions from the fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Knnr, X)
    Q = eltype(object.X)
    X = ensure_mat(X)
    m = nro(X)
    q = nco(object.Y)
    ## Getknn
    metric = object.par.metric
    h = object.par.h
    k = object.par.k
    tolw = object.par.tolw
    criw = object.par.criw
    squared = object.par.squared
    if isnothing(object.fm)
        if object.par.scal
            zX1 = fscale(object.X, object.xscales)
            zX2 = fscale(X, object.xscales)
            res = getknn(zX1, zX2; metric, k)
        else
            res = getknn(object.X, X; metric, k)
        end
    else
        res = getknn(object.fm.T, transf(object.fm, X); metric, k) 
    end
    listw = copy(res.d)
    Threads.@threads for i = 1:m
        w = wdist(res.d[i]; h, criw, squared)
        w[w .< tolw] .= tolw
        listw[i] = w
    end
    ## End
    pred = zeros(Q, m, q)
    @inbounds for i = 1:m
        weights = mweight(listw[i])
        pred[i, :] .= colmean(vrow(object.Y, res.ind[i]), weights)
    end
    (pred = pred, listnn = res.ind, listd = res.d, listw = listw)
end

