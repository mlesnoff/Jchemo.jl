"""
    lwmlr(; kwargs...)
    lwmlr(X, Y; kwargs...)
k-Nearest-Neighbours locally weighted multiple linear regression (kNN-LWMLR).
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
Keyword arguments:
* `metric` : Type of dissimilarity used to select the neighbors and to compute the weights 
    (see function `getknn`). Possible values are: `:eucl` (Euclidean), `:mah` (Mahalanobis), 
    `:sam` (spectral angular distance), `:cos` (cosine distance), `:cor` (correlation distance).
* `k` : The number of nearest neighbors to select for each observation to predict.
* `h` : A scalar defining the shape of the weight function computed by function `winvs`. Lower is h, 
    sharper is the function. See function `winvs` for details (keyword arguments `criw` and `squared` of 
    `winvs` can also be specified here).
* `tolw` : For stabilization when very close neighbors.
* `scal` : Symbol defining the column scaling of the global `X` (before the computation of the distances and local weights). 
    Possible values are: `:none`, `std` (uncorrected STD), `prt` (pareto) and `:mad` (MAD).
* `store` : Boolean. If `true`, the local models fitted on the neighborhoods are stored and returned by function `predict`.
* `verbose` : Boolean. If `true`, predicting information are printed.
    
This is the same principle as function `lwplsr` except that MLR models are fitted on the neighborhoods, instead of 
PLSR models.  The neighborhoods are computed directly on `X` (there is no preliminary dimension reduction).

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

nlv = 20
model0 = pcasvd(; nlv) ;
fit!(model0, Xtrain) 
@head Ttrain = model0.fitm.T 
@head Ttest = transf(model0, Xtest)

metric = :eucl 
h = 2. ; k = 100 
model = lwmlr(; metric, h, k) 
fit!(model, Ttrain, ytrain)
@names model
@names model.fitm

res = predict(model, Ttest) ; 
@names res 
@head res.listnn
@head res.listd
@head res.listw
@head res.pred
@show rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction",  
    ylabel = "Observed").f    

## Same but with function 'pip'
nlv = 20
metric = :eucl 
h = 2. ; k = 100 
model1 = pcasvd(; nlv) ;
model2 = lwmlr(; metric, h, k) 
model = pip(model1, model2)
fit!(model, Xtrain, ytrain)
res = predict(model, Xtest) ;
@head res.pred
rmsep(res.pred, ytest)

####### Example of fitting the function sinc(x) described in Rosipal & Trejo 2001 p. 105-106 
 
x = collect(-10:.2:10) 
x[x .== 0] .= 1e-5
n = length(x)
zy = sin.(abs.(x)) ./ abs.(x) 
y = zy + .2 * randn(n) 
model = lwmlr(metric = :eucl, h = 1.5, k = 20) ;
fit!(model, x, y)
pred = predict(model, x).pred 
f, ax = scatter(x, y) 
lines!(ax, x, zy, label = "True model")
lines!(ax, x, vec(pred), label = "Fitted model")
axislegend("Method")
f
```
""" 
lwmlr(; kwargs...) = JchemoModel(lwmlr, nothing, kwargs)

function lwmlr(X, Y; kwargs...) 
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    p = nco(X)
    Q = eltype(X) 
    par = recovkw(ParLwmlr{Q}, kwargs).par
    xscales = ones(Q, p)
    if par.scal != :none
        colscal = def_colscal(par.scal) 
        xscales .= colscal(X, weights)
        X = fscale(X, xscales)
    end
    Lwmlr(X, Y, xscales, par)
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
    Q = eltype(object.X)
    ## Getknn
    metric = object.par.metric
    k = object.par.k
    h = object.par.h
    criw = object.par.criw
    squared = object.par.squared
    tolw = object.par.tolw
    if object.par.scal != :none
        zX1 = fscale(object.X, object.xscales)
        zX2 = fscale(X, object.xscales)
        res = getknn(zX1, zX2; metric, k)
    else
        res = getknn(object.X, X; metric, k)
    end
    listw = similar(res.d)
    Threads.@threads for i in eachindex(res.d)
        w = winvs(res.d[i]; h, criw, squared)
        @. w[w < tolw] = tolw
        listw[i] = w
    end
    ## End
    reslocw = locw(object.X, object.Y, X; listnn = res.ind, listw, algo = mlr, store = object.par.store, 
        verbose = object.par.verbose)
    (pred = reslocw.pred, fitm = reslocw.fitm, listnn = res.ind, listd = res.d, listw)
end

