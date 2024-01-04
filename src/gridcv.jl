"""
    gridcv(mod, X, Y; segm, score, 
        pars = nothing, nlv = nothing, lb = nothing, 
        verbose = false)
Cross-validation (CV) of a model over a grid of parameters.
* `mod` : Model to evaluate.
* `X` : Training X-data (n, p).
* `Y` : Training Y-data (n, q).
Keyword arguments: 
* `segm` : Segments of observations used for 
    the CV (output of functions [`segmts`](@ref), 
    [`segmkf`](@ref), etc.).
* `score` : Function computing the prediction 
    score (e.g. `rmsep`).
* `pars` : tuple of named vectors of same length defining 
    the parameter combinations (e.g. output of function `mpar`).
* `verbose` : If `true`, fitting information are printed.
* `nlv` : Value, or vector of values, of the nb. of latent
    variables (LVs).
* `lb` : Value, or vector of values, of the ridge 
    regularization parameter "lambda".

The function is used for grid-search: it computed a prediction score 
(= error rate) for model `mod` over the combinations of parameters 
defined in `pars`. 
    
For models based on LV or ridge regularization, using arguments `nlv` 
and `lb` allow faster computations than including these parameters in 
argument `pars. See the examples.   

The function returns two outputs: 
* `res` : mean results
* `res_p` : results per replication.

## Examples
```julia
######## Regression

using JLD2, CairoMakie, JchemoData
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "cassav.jld2") 
@load db dat
pnames(dat)
X = dat.X 
y = dat.Y.tbc
year = dat.Y.year
tab(year)
mod = savgol(npoint = 21, 
    deriv = 2, degree = 2)
fit!(mod, X)
Xp = transf(mod, X)
s = year .<= 2012
Xtrain = Xp[s, :]
ytrain = y[s]
Xtest = rmrow(Xp, s)
ytest = rmrow(y, s)
ntrain = nro(Xtrain)
ntest = nro(Xtest)
ntot = ntrain + ntest
(ntot = ntot, ntrain, ntest)

## Replicated K-fold CV 
K = 3 ; rep = 10
segm = segmkf(ntrain, K; rep)
## Replicated test-set validation
#m = Int(round(ntrain / 3)) ; rep = 30
#segm = segmts(ntrain, m; rep)

####-- Plsr
mod = plskern()
nlv = 0:30
rescv = gridcv(mod, Xtrain, ytrain; 
    segm, score = rmsep, nlv) ;
pnames(rescv)
res = rescv.res 
plotgrid(res.nlv, res.y1; step = 2,
    xlabel = "Nb. LVs", ylabel = "RMSEP").f
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
mod = plskern(nlv = res.nlv[u])
fit!(mod, Xtrain, ytrain)
pred = predict(mod, Xtest).pred
@show rmsep(pred, ytest)
plotxy(vec(pred), ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f    

## Adding pars 
pars = mpar(scal = [false; true])
rescv = gridcv(mod, Xtrain, ytrain; 
    segm, score = rmsep, pars, nlv) ;
res = rescv.res 
typ = res.scal
plotgrid(res.nlv, res.y1, typ; step = 2,
    xlabel = "Nb. LVs", ylabel = "RMSEP").f
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
mod = plskern(nlv = res.nlv[u], 
    scal = res.scal[u])
fit!(mod, Xtrain, ytrain)
pred = predict(mod, Xtest).pred
@show rmsep(pred, ytest)
plotxy(vec(pred), ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f    

####-- Rr 
lb = (10).^(-8:.1:3)
mod = rr() 
rescv = gridcv(mod, Xtrain, ytrain; 
    segm, score = rmsep, lb) ;
res = rescv.res 
loglb = log.(10, res.lb)
plotgrid(loglb, res.y1; step = 2,
    xlabel = "log(lambda)", ylabel = "RMSEP").f
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
mod = rr(lb = res.lb[u])
fit!(mod, Xtrain, ytrain)
pred = predict(mod, Xtest).pred
@show rmsep(pred, ytest)
plotxy(vec(pred), ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f    
    
## Adding pars 
pars = mpar(scal = [false; true])
rescv = gridcv(mod, Xtrain, ytrain; 
    segm, score = rmsep, pars, lb) ;
res = rescv.res 
loglb = log.(10, res.lb)
typ = string.(res.scal)
plotgrid(loglb, res.y1, typ; step = 2,
    xlabel = "log(lambda)", ylabel = "RMSEP").f
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
mod = rr(lb = res.lb[u],
    scal = res.scal[u])
fit!(mod, Xtrain, ytrain)
pred = predict(mod, Xtest).pred
@show rmsep(pred, ytest)
plotxy(vec(pred), ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f    

####-- Kplsr 
mod = kplsr()
nlv = 0:30
gamma = (10).^collect(-5:1.:5)
pars = mpar(gamma = gamma)
rescv = gridcv(mod, Xtrain, ytrain; 
    segm, score = rmsep, pars, nlv) ;
res = rescv.res 
loggamma = round.(log.(10, res.gamma), digits = 1)
plotgrid(res.nlv, res.y1, loggamma; step = 2,
    xlabel = "Nb. LVs", ylabel = "RMSEP",
    leg_title = "Log(gamma)").f
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
mod = kplsr(nlv = res.nlv[u], 
    gamma = res.gamma[u])
fit!(mod, Xtrain, ytrain)
pred = predict(mod, Xtest).pred
@show rmsep(pred, ytest)
plotxy(vec(pred), ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f    

####-- Knnr 
nlvdis = [15, 25] ; metric = [:mah]
h = [1, 2.5, 5]
k = [1; 5; 10; 20; 50 ; 100] 
pars = mpar(nlvdis = nlvdis, 
    metric = metric, h = h, k = k)
length(pars[1]) 
mod = knnr()
rescv = gridcv(mod, Xtrain, ytrain;
    segm, score = rmsep, pars, 
    verbose = true) ;
res = rescv.res 
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
mod = knnr(nlvdis = res.nlvdis[u],
    metric = res.metric[u], h = res.h[u], 
    k = res.k[u])
fit!(mod, Xtrain, ytrain)
pred = predict(mod, Xtest).pred
@show rmsep(pred, ytest)
plotxy(vec(pred), ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f    

####-- Lwplsr 
nlvdis = 15 ; metric = [:mah]
h = [1, 2.5, 5] ; k = [50, 100] 
pars = mpar(nlvdis = nlvdis, 
    metric = metric, h = h, k = k)
length(pars[1]) 
nlv = 0:20
mod = lwplsr()
rescv = gridcv(mod, Xtrain, ytrain;
    segm, score = rmsep, pars, nlv, 
    verbose = true) ;
res = rescv.res 
group = string.("h=", res.h, " k=", res.k)
plotgrid(res.nlv, res.y1, group;
    xlabel = "Nb. LVs", ylabel = "RMSEP").f
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
mod = lwplsr(nlvdis = res.nlvdis[u],
    metric = res.metric[u], h = res.h[u], 
    k = res.k[u], nlv = res.nlv[u])
fit!(mod, Xtrain, ytrain)
pred = predict(mod, Xtest).pred
@show rmsep(pred, ytest)
plotxy(vec(pred), ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f    

####-- LwplsrAvg 
nlvdis = 15 ; metric = [:mah]
h = [1, 2.5, 5] ; k = [50, 100]
nlv = [0:15, 0:20, 5:20]  
pars = mpar(nlvdis = nlvdis, 
    metric = metric, h = h, k = k,
    nlv = nlv)
length(pars[1]) 
mod = lwplsravg()
rescv = gridcv(mod, Xtrain, ytrain;
    segm, score = rmsep, pars, 
    verbose = true) ;
res = rescv.res 
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
mod = lwplsr(nlvdis = res.nlvdis[u],
    metric = res.metric[u], h = res.h[u], 
    k = res.k[u], nlv = res.nlv[u])
fit!(mod, Xtrain, ytrain)
pred = predict(mod, Xtest).pred
@show rmsep(pred, ytest)
plotxy(vec(pred), ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f    

######## Discrimination
## The principle is the same as 
## for regression

using JLD2, CairoMakie, JchemoData
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/forages2.jld2")
@load db dat
pnames(dat)
X = dat.X
Y = dat.Y
tab(Y.typ)
s = Bool.(Y.test)
Xtrain = rmrow(X, s)
ytrain = rmrow(Y.typ, s)
Xtest = X[s, :]
ytest = Y.typ[s]
ntrain = nro(Xtrain)
ntest = nro(Xtest)
ntot = ntrain + ntest
(ntot = ntot, ntrain, ntest)

## Replicated K-fold CV 
K = 3 ; rep = 10
segm = segmkf(ntrain, K; rep)
## Replicated test-set validation
#m = Int(round(ntrain / 3)) ; rep = 30
#segm = segmts(ntrain, m; rep)

####-- Plslda
mod = plslda()
nlv = 1:30
prior = [:unif; :prop]
pars = mpar(prior = prior)
rescv = gridcv(mod, Xtrain, ytrain; 
    segm, score = errp, pars, nlv)
res = rescv.res
typ = res.prior
plotgrid(res.nlv, res.y1, typ; step = 2,
    xlabel = "Nb. LVs", ylabel = "ERR").f
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
mod = plslda(nlv = res.nlv[u], 
    prior = res.prior[u])
fit!(mod, Xtrain, ytrain)
pred = predict(mod, Xtest).pred
@show errp(pred, ytest)
confusion(pred, ytest).pct
```
"""
function gridcv(mod, X, Y; segm, score, 
        pars = nothing, nlv = nothing, lb = nothing, 
        verbose = false)
    fun = mod.fun 
    if isnothing(nlv) && isnothing(lb)
        res = gridcv_br(X, Y; segm, fun, score, 
            pars, verbose)
    elseif !isnothing(nlv)
        res = gridcv_lv(X, Y; segm, fun, score, 
            pars, nlv, verbose)
    elseif !isnothing(lb)
        res = gridcv_lb(X, Y; segm, fun, score, 
            pars, lb, verbose)
    end
    res
end
    
