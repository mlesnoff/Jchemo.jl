"""
    lwmlr(X, Y; kwargs...)
k-Nearest-Neighbours locally weighted multiple linear 
    regression (kNN-LWMLR).
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
Keyword arguments:
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

This is the same principle as function `lwplsr` except 
that MLR models are fitted on the neighborhoods, instead of 
PLSR models.  The neighborhoods are computed directly on `X` 
(there is no preliminary dimension reduction).

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

nlv = 20
mod0 = model(pcasvd; nlv) ;
fit!(mod0, Xtrain) 
@head Ttrain = mod0.fm.T 
@head Ttest = transf(mod0, Xtest)

metric = :eucl 
h = 2 ; k = 100 
mod = model(lwmlr; metric, h, k) 
fit!(mod, Ttrain, ytrain)
pnames(mod)
pnames(mod.fm)

res = predict(mod, Ttest) ; 
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
mod = model(lwmlr; metric = :eucl, h = 1.5, k = 20) ;
fit!(mod, x, y)
pred = predict(mod, x).pred 
f, ax = scatter(x, y) 
lines!(ax, x, zy, label = "True model")
lines!(ax, x, vec(pred), label = "Fitted model")
axislegend("Method")
f
```
""" 
function lwmlr(X, Y; kwargs...) 
    par = recovkwargs(Par, kwargs)
    X = ensure_mat(X)  
    Y = ensure_mat(Y)
    Lwmlr(X, Y, kwargs, par)
end

"""
    predict(object::Lwmlr, X)
Compute the Y-predictions from the fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Lwmlr, X)
    X = ensure_mat(X)
    m = nro(X)
    ## Getknn
    metric = object.par.metric
    h = object.par.h
    k = object.par.k
    tolw = object.par.tolw
    criw = object.par.criw
    squared = object.par.squared
    res = getknn(object.X, X; metric, k)
    listw = copy(res.d)
    Threads.@threads for i = 1:m
        w = wdist(res.d[i]; h, criw, squared)
        w[w .< tolw] .= tolw
        listw[i] = w
    end
    ## End
    pred = locw(object.X, object.Y, X; listnn = res.ind, listw, 
        fun = mlr, verbose = object.par.verbose).pred
    (pred = pred, listnn = res.ind, listd = res.d, listw)
end

