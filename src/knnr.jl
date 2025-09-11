"""
    knnr(; kwargs...)
    knnr(X, Y; kwargs...) 
k-Nearest-Neighbours weighted regression (KNNR).
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
Keyword arguments:
* `metric` : Type of dissimilarity used to select the neighbors and to compute the weights 
    (see function `getknn`). Possible values are: `:eucl` (Euclidean), `:mah` (Mahalanobis), 
    `:sam` (a spectral angular distance), `:cos` (cosine distance), `:cor` (a correlation distance).
* `h` : A scalar defining the shape of the weight function computed by function `winvs`. Lower is h, 
    sharper is the function. See function `winvs` for details (keyword arguments `criw` and `squared` of 
    `winvs` can also be specified here).
* `k` : The number of nearest neighbors to select for each observation to predict.
* `tolw` : For stabilization when very close neighbors.
* `scal` : Boolean. If `true`, each column of the global `X` is scaled by its uncorrected standard 
    deviation before the distance and weight computations.

The general principle of this function is as follows (many other variants of kNNR pipelines can be built):
a) For each new observation to predict, the prediction is the weighted mean of `y` over a selected neighborhood
    (in `X`) of size `k`. 
b) Within the selected neighborhood, the weights  are defined from the dissimilarities between the new observation 
    and the neighborhood, and are computed from function 'winvs'.
    
In general, for X-data with high dimensions, using the Mahalanobis distance requires a preliminary dimensionality 
reduction (see examples).

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
@names dat
X = dat.X 
y = dat.Y.tbc
year = dat.Y.year
tab(year)
s = year .<= 2012
Xtrain = X[s, :]
ytrain = y[s]
Xtest = rmrow(X, s)
ytest = rmrow(y, s)

h = 1 ; k = 3 
model = knnr(; h, k) 
fit!(model, Xtrain, ytrain)
@names model
@names model.fitm
dump(model.fitm.par)
res = predict(model, Xtest) ; 
@names res 
res.listnn
res.listd
res.listw
@head res.pred
@show rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f    

## With dimension reduction
model1 = pcasvd(nlv = 15)
metric = :eucl ; h = 1 ; k = 3 
model2 = knnr(; metric, h, k) 
model = pip(model1, model2)
fit!(model, Xtrain, ytrain)
res = predict(model, Xtest) ; 
@head res.pred
@show rmsep(res.pred, ytest)

####### Example of fitting the function sinc(x)
####### described in Rosipal & Trejo 2001 p. 105-106 
x = collect(-10:.2:10) 
x[x .== 0] .= 1e-5
n = length(x)
zy = sin.(abs.(x)) ./ abs.(x) 
y = zy + .2 * randn(n) 
model = knnr(k = 15, h = 5) 
fit!(model, x, y)
pred = predict(model, x).pred 
f, ax = scatter(x, y) 
lines!(ax, x, zy, label = "True model")
lines!(ax, x, vec(pred), label = "Fitted model")
axislegend("Method")
f
```
""" 
knnr(; kwargs...) = JchemoModel(knnr, nothing, kwargs)

function knnr(X, Y; kwargs...) 
    par = recovkw(ParKnn, kwargs).par
    @assert in([:eucl, :mah])(par.metric) "Wrong value for argument 'metric'."
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    Q = eltype(X)
    p = nco(X)
    xscales = ones(Q, p)
    if par.scal
        xscales .= colstd(X)
    end
    Knnr(X, Y, xscales, par) 
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
    if object.par.scal
        zX1 = fscale(object.X, object.xscales)
        zX2 = fscale(X, object.xscales)
        res = getknn(zX1, zX2; metric, k)
    else
        res = getknn(object.X, X; metric, k)
    end
    listw = copy(res.d)
    Threads.@threads for i = 1:m
        w = winvs(res.d[i]; h, criw, squared)
        w[w .< tolw] .= tolw
        listw[i] = w
    end
    ## End
    pred = zeros(Q, m, q)
    @inbounds for i = 1:m
        weights = mweight(listw[i])
        pred[i, :] .= colmean(vrow(object.Y, res.ind[i]), weights)
    end
    (pred = pred, listnn = res.ind, listd = res.d, listw)
end

