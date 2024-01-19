"""
    gridscore(mod, Xtrain, Ytrain, X, Y; score, pars = nothing, 
        nlv = nothing, lb = nothing, verbose = false) 
Test-set validation of a model over a grid of parameters.
* `mod` : Model to evaluate.
* `Xtrain` : Training X-data (n, p).
* `Ytrain` : Training Y-data (n, q).
* `X` : Validation X-data (m, p).
* `Y` : Validation Y-data (m, q).
Keyword arguments: 
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
defined in `pars`. The score is computed over sets {`X, `Y`}. 
    
For models based on LV or ridge regularization, using arguments `nlv` 
and `lb` allow faster computations than including these parameters in 
argument `pars. See the examples.   

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
## Building Cal and Val 
## within Train
nval = Int(round(.3 * ntrain))
s = samprand(ntrain, nval)
Xcal = Xtrain[s.train, :]
ycal = ytrain[s.train]
Xval = Xtrain[s.test, :]
yval = ytrain[s.test]

####-- Plsr
mod = plskern()
nlv = 0:30
res = gridscore(mod, Xcal, ycal, Xval, yval; 
    score = rmsep, nlv)
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
res = gridscore(mod, Xcal, ycal, Xval, yval; 
    score = rmsep, pars = pars, nlv = nlv)
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
res = gridscore(mod, Xcal, ycal, Xval, yval; 
    score = rmsep, lb)
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
res = gridscore(mod, Xcal, ycal, Xval, yval; 
    score = rmsep, pars, lb)
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
gamma = (10).^(-5:1.:5)
pars = mpar(gamma = gamma)
res = gridscore(mod, Xcal, ycal, Xval, yval; 
    score = rmsep, pars = pars, nlv = nlv)
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
nlvdis = [15; 25] ; metric = [:mah]
h = [1, 2.5, 5]
k = [1, 5, 10, 20, 50, 100] 
pars = mpar(nlvdis = nlvdis, 
    metric = metric, h = h, k = k)
length(pars[1]) 
mod = knnr()
res = gridscore(mod, Xcal, ycal, Xval, yval;
    score = rmsep, pars, verbose = true)
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
res = gridscore(mod, Xcal, ycal, Xval, yval;
    score = rmsep, pars, nlv, verbose = true)
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
res = gridscore(mod, Xcal, ycal, Xval, yval;
    score = rmsep, pars, verbose = true)
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
mod = lwplsravg(nlvdis = res.nlvdis[u],
    metric = res.metric[u], h = res.h[u], 
    k = res.k[u], nlv = res.nlv[u])
fit!(mod, Xtrain, ytrain)
pred = predict(mod, Xtest).pred
@show rmsep(pred, ytest)
plotxy(vec(pred), ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f   

####-- Mbplsr
listbl = [1:525, 526:1050]
Xbltrain = mblock(Xtrain, listbl)
Xbltest = mblock(Xtest, listbl) 
Xbl_cal = mblock(Xcal, listbl) 
Xbl_val = mblock(Xval, listbl) 

mod = mbplsr()
bscal = [:none, :frob]
pars = mpar(bscal = bscal) 
nlv = 0:30
res = gridscore(mod, Xbl_cal, ycal, Xbl_val, yval; 
    score = rmsep, pars, nlv)
group = res.bscal 
plotgrid(res.nlv, res.y1, group; step = 2,
    xlabel = "Nb. LVs", ylabel = "RMSEP").f
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
mod = mbplsr(bscal = res.bscal[u], 
    nlv = res.nlv[u])
fit!(mod, Xbltrain, ytrain)
pred = predict(mod, Xbltest).pred
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
## Building Cal and Val 
## within Train
nval = Int(round(.3 * ntrain))
s = samprand(ntrain, nval)
Xcal = Xtrain[s.train, :]
ycal = ytrain[s.train]
Xval = Xtrain[s.test, :]
yval = ytrain[s.test]

####-- Plslda
mod = plslda()
nlv = 1:30
prior = [:unif, :prop]
pars = mpar(prior = prior)
res = gridscore(mod, Xcal, ycal, Xval, yval; 
    score = errp, pars, nlv)
typ = res.prior
plotgrid(res.nlv, res.y1, typ; step = 2,
    xlabel = "Nb. LVs", ylabel = "RMSEP").f
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
function gridscore(mod, Xtrain, Ytrain, X, Y; score, pars = nothing, 
        nlv = nothing, lb = nothing, verbose = false)
    fun = mod.fun
    if isnothing(nlv) && isnothing(lb)
        res = gridscore_br(Xtrain, Ytrain, X, Y; fun, score, pars, 
            verbose)
    elseif !isnothing(nlv)
        res = gridscore_lv(Xtrain, Ytrain, X, Y; fun, score, pars, 
            nlv, verbose)
    elseif !isnothing(lb)
        res = gridscore_lb(Xtrain, Ytrain, X, Y; fun, score, pars, 
            lb, verbose)
    end
    res
end

