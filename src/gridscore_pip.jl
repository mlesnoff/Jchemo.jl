"""
    gridscore(mod::Pipeline, Xtrain, Ytrain, X, Y; score, pars = nothing, 
        nlv = nothing, lb = nothing, verbose = false) 
Test-set validation of a model pipeline over a grid of parameters.
* `mod` : A pipeline of models to evaluate.
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

In the present version of the function, only the last model 
of the pipeline (= the final predictor) is validated.

For other details, see function `gridscore` for simple models. 

## Examples
```julia
using JLD2, CairoMakie, JchemoData
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "cassav.jld2") 
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

####-- Snv :> Savgol :> Plsr
## Only the last model is validated
## mod1
centr = true ; scal = false
mod1 = model(snv; centr, scal)
## mod2 
npoint = 11 ; deriv = 2 ; degree = 3
mod2 = model(savgol; npoint, deriv, degree)
## mod3
nlv = 0:30
mod3 = model(plskern)
##
mod = pip(mod1, mod2, mod3)
res = gridscore(mod, Xcal, ycal, Xval, yval; score = rmsep, nlv) ;
plotgrid(res.nlv, res.y1; step = 2, xlabel = "Nb. LVs", ylabel = "RMSEP").f
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
mod3 = model(plskern; nlv = res.nlv[u])
mod = pip(mod1, mod2, mod3)
fit!(mod, Xtrain, ytrain)
res = predict(mod, Xtest) ; 
@head res.pred 
rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction",
      ylabel = "Observed").f

####-- Pca :> Svmr
## Only the last model is validated
## mod1
nlv = 15 ; scal = true
mod1 = model(pcasvd; nlv, scal)
## mod2
kern = [:krbf]
gamma = (10).^(-5:1.:5)
cost = (10).^(1:3)
epsilon = [.1, .2, .5]
pars = mpar(kern = kern, gamma = gamma, cost = cost, epsilon = epsilon)
mod2 = model(svmr)
##
mod = pip(mod1, mod2)
res = gridscore(mod, Xcal, ycal, Xval, yval; score = rmsep, pars)
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
mod2 = model(svmr; kern = res.kern[u], gamma = res.gamma[u], cost = res.cost[u],
    epsilon = res.epsilon[u])
mod = pip(mod1, mod2) ;
fit!(mod, Xtrain, ytrain)
res = predict(mod, Xtest) ; 
@head res.pred 
rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction",
      ylabel = "Observed").f
```
"""
function gridscore(mod::Pipeline, Xtrain, Ytrain, X, Y; score, pars = nothing, 
        nlv = nothing, lb = nothing, verbose = false)
    fit!(mod, Xtrain, Ytrain)
    K = length(mod.mod)
    for i = 1:(K - 1)
        Xtrain = transf(mod.mod[i], Xtrain)
        X = transf(mod.mod[i], X)
    end
    gridscore(mod.mod[K], Xtrain, Ytrain, X, Y; score, pars, nlv, lb, verbose)
end

