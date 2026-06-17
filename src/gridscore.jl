"""
    gridscore(model, Xtrain, Ytrain, X, Y; score::Function, pars::Union{Nothing, NamedTuple} = nothing, 
        nlv::Union{Nothing, Int, AbstractVector{Int}} = nothing, 
        lb::Union{Nothing, Float64, AbstractVector{Float64}} = nothing, 
        verbose::Bool = false)
    gridscore(model::Pipeline, Xtrain, Ytrain, X, Y; score::Function, 
        pars::Union{Nothing, NamedTuple} = nothing, 
        nlv::Union{Nothing, Int, AbstractVector{Int}} = nothing, 
        lb::Union{Nothing, Float64, AbstractVector{Float64}} = nothing, 
        verbose::Bool = false) 
Test-set validation of a model over a grid of parameters.
* `model` : Model to evaluate.
* `Xtrain` : Training X-data (n, p).
* `Ytrain` : Training Y-data (n, q).
* `X` : Validation X-data (m, p).
* `Y` : Validation Y-data (m, q).
Keyword arguments: 
* `score` : Function computing the prediction score (e.g., `rmsep`).
* `pars` : tuple of named vectors of same length defining the parameter combinations (e.g., output 
    of function `mpar`).
* `verbose` : If `true`, predicting information are printed.
* `nlv` : Value, or vector of values, of the nb. of latent variables (LVs).
* `lb` : Value, or vector of values, of the ridge regularization parameter "lambda".

The function is used for grid-search: it computes a prediction score (= error rate) for the specified `model` 
for each parameter combination defined in `pars`. The score is computed over sets {`X, `Y`}. 
    
For models based on LV or ridge regularization, using arguments `nlv` and `lb` allow faster computations than including 
these parameters in argument `pars. See the examples.   

**For pipeline models:** In the present version of the function, only the last model of the pipeline 
(= the final predictor) is tuned. Therefore, argument `pars` must only contain parameters for this last model.

## Examples
```julia
####### Regression 

using Jchemo, JLD2, CairoMakie, JchemoData
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "cassav.jld2") 
@load db dat
@names dat
X = dat.X 
y = dat.Y.tbc
year = dat.Y.year
tab(year)
model = savgol(npoint = 21, deriv = 2, degree = 2)
fit!(model, X)
Xp = transf(model, X)
s = year .<= 2012
Xtrain = Xp[s, :]
ytrain = y[s]
Xtest = rmrow(Xp, s)
ytest = rmrow(y, s)
ntrain = nro(Xtrain)
ntest = nro(Xtest)
ntot = ntrain + ntest
(ntot = ntot, ntrain, ntest)
## Train ==> Cal + Val 
nval = round(Int, .3 * ntrain)
s = samprand(ntrain, nval; seed = 1234)
Xcal = Xtrain[s.train, :]
ycal = ytrain[s.train]
Xval = Xtrain[s.test, :]
yval = ytrain[s.test]

####---- Plsr
model = plskern()
nlv = 0:30
res = gridscore(model, Xcal, ycal, Xval, yval; score = rmsep, nlv)
plotgrid(res.nlv, res.y1; step = 2, xlabel = "Nb. LVs", ylabel = "RMSEP").f
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
model = plskern(nlv = res.nlv[u])
fit!(model, Xtrain, ytrain)
pred = predict(model, Xtest).pred
@show rmsep(pred, ytest)
plotxy(vec(pred), ytest; color = (:red, .5), bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f    

## Adding pars 
pars = mpar(scal = [false; true])
res = gridscore(model, Xcal, ycal, Xval, yval; score = rmsep, pars, nlv)
typ = res.scal
plotgrid(res.nlv, res.y1, typ; step = 2, xlabel = "Nb. LVs", ylabel = "RMSEP").f
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
model = plskern(nlv = res.nlv[u], scal = res.scal[u])
fit!(model, Xtrain, ytrain)
pred = predict(model, Xtest).pred
@show rmsep(pred, ytest)
plotxy(vec(pred), ytest; color = (:red, .5), bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f    

####---- Rr 
lb = (10).^(-8:.1:3)
model = rr() 
res = gridscore(model, Xcal, ycal, Xval, yval; score = rmsep, lb)
loglb = log.(10, res.lb)
plotgrid(loglb, res.y1; step = 2, xlabel = "log(lambda)", ylabel = "RMSEP").f
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
model = rr(lb = res.lb[u])
fit!(model, Xtrain, ytrain)
pred = predict(model, Xtest).pred
@show rmsep(pred, ytest)
plotxy(vec(pred), ytest; color = (:red, .5), bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f    
    
## Adding pars 
pars = mpar(scal = [false; true])
res = gridscore(model, Xcal, ycal, Xval, yval; score = rmsep, pars, lb)
loglb = log.(10, res.lb)
typ = string.(res.scal)
plotgrid(loglb, res.y1, typ; step = 2, xlabel = "log(lambda)", ylabel = "RMSEP").f
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
model = rr(lb = res.lb[u], scal = res.scal[u])
fit!(model, Xtrain, ytrain)
pred = predict(model, Xtest).pred
@show rmsep(pred, ytest)
plotxy(vec(pred), ytest; color = (:red, .5), bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f    

####---- Kplsr 
model = kplsr()
nlv = 0:30
gamma = (10).^(-5:1.:5)
pars = mpar(gamma = gamma)
res = gridscore(model, Xcal, ycal, Xval, yval; score = rmsep, pars, nlv)
lgamma = round.(log.(10, res.gamma), digits = 1)
plotgrid(res.nlv, res.y1, lgamma; step = 2, xlabel = "Nb. LVs", ylabel = "RMSEP",
    leg_title = "Log(gamma)").f
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
model = kplsr(nlv = res.nlv[u], gamma = res.gamma[u])
fit!(model, Xtrain, ytrain)
pred = predict(model, Xtest).pred
@show rmsep(pred, ytest)
plotxy(vec(pred), ytest; color = (:red, .5), bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f    

####---- Knnr 
nlvdis = [0; 15; 25] ; metric = [:cos]
h = [1, 2.5, 5]
k = [1, 5, 10, 20, 50, 100] 
pars = mpar(nlvdis = nlvdis, metric = metric, h = h, k = k)
length(pars[1]) 
model = knnr()
res = gridscore(model, Xcal, ycal, Xval, yval; score = rmsep, pars, verbose = true)
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
model = knnr(nlvdis = res.nlvdis[u], metric = res.metric[u], h = res.h[u], k = res.k[u])
fit!(model, Xtrain, ytrain)
pred = predict(model, Xtest).pred
@show rmsep(pred, ytest)
plotxy(vec(pred), ytest; color = (:red, .5), bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f    

####---- Lwplsr 
nlvdis = 15 ; metric = [:mah]
h = [1, 2, 5] ; k = [50, 75, 100] 
pars = mpar(nlvdis = nlvdis, metric = metric, h = h, k = k)
length(pars[1]) 
nlv = 0:20
model = lwplsr()
res = gridscore(model, Xcal, ycal, Xval, yval; score = rmsep, pars, nlv, verbose = true)
group = string.("h=", res.h, " k=", res.k)
plotgrid(res.nlv, res.y1, group; xlabel = "Nb. LVs", ylabel = "RMSEP").f
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
model = lwplsr(nlvdis = res.nlvdis[u], metric = res.metric[u], h = res.h[u], 
    k = res.k[u], nlv = res.nlv[u])
fit!(model, Xtrain, ytrain)
pred = predict(model, Xtest).pred
@show rmsep(pred, ytest)
plotxy(vec(pred), ytest; color = (:red, .5), bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f    

####---- LwplsrAvg 
nlvdis = 15 ; metric = [:mah]
h = [1, 2, 5] ; k = [50, 75, 100] 
nlv = [0:20, 5:20] 
pars = mpar(nlvdis = nlvdis, metric = metric, h = h, k = k, nlv = nlv)
length(pars[1]) 
model = lwplsravg()
res = gridscore(model, Xcal, ycal, Xval, yval; score = rmsep, pars, verbose = true)
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
model = lwplsravg(nlvdis = res.nlvdis[u], metric = res.metric[u], h = res.h[u], 
    k = res.k[u], nlv = res.nlv[u])
fit!(model, Xtrain, ytrain)
pred = predict(model, Xtest).pred
@show rmsep(pred, ytest)
plotxy(vec(pred), ytest; color = (:red, .5), bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f   

####---- Mbplsr
listbl = [1:525, 526:1050]
Xbltrain = mblock(Xtrain, listbl)
Xbltest = mblock(Xtest, listbl) 
Xblcal = mblock(Xcal, listbl) 
Xblval = mblock(Xval, listbl) 

model = mbplsr()
bscal = [:none, :frob]
pars = mpar(bscal = bscal) 
nlv = 0:30
res = gridscore(model, Xblcal, ycal, Xblval, yval; score = rmsep, pars, nlv)
group = res.bscal 
plotgrid(res.nlv, res.y1, group; step = 2, xlabel = "Nb. LVs", ylabel = "RMSEP").f
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
model = mbplsr(bscal = res.bscal[u], nlv = res.nlv[u])
fit!(model, Xbltrain, ytrain)
pred = predict(model, Xbltest).pred
@show rmsep(pred, ytest)
plotxy(vec(pred), ytest; color = (:red, .5), bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f    

## Pipelines

####-- Pipeline Snv :> Savgol :> Plsr   (Only the last model is tuned)
## model1
model1 = snv(scal = false)
## model2 
npoint = 5 ; deriv = 0 ; degree = 2
model2 = savgol(; npoint, deriv, degree)
## model3
nlv = 0:30
model3 = plskern()
## Pipeline
model = pip(model1, model2, model3)
res = gridscore(model, Xcal, ycal, Xval, yval; score = rmsep, nlv) ;
@head res
plotgrid(res.nlv, res.y1; step = 2, xlabel = "Nb. LVs", ylabel = "RMSEP").f
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
model3 = plskern(nlv = res.nlv[u])
model = pip(model1, model2, model3)
fit!(model, Xtrain, ytrain)
res = predict(model, Xtest) ; 
@head res.pred 
rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction",
      ylabel = "Observed").f

####-- Pipeline Pca :> Svmr   (Only the last model is tuned)
## model1
nlv = 15 ; scal = :std
model1 = pcasvd(; nlv, scal)
## model2
kern = [:krbf]
gamma = (10).^(-5:1.:5)
cost = (10).^(1:3)
epsilon = [.1, .2, .5]
pars = mpar(kern = kern, gamma = gamma, cost = cost, epsilon = epsilon)
model2 = svmr()
## Pipeline
model = pip(model1, model2)
res = gridscore(model, Xcal, ycal, Xval, yval; score = rmsep, pars, verbose = true)
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
model2 = svmr(kern = res.kern[u], gamma = res.gamma[u], cost = res.cost[u], epsilon = res.epsilon[u])
model = pip(model1, model2) 
fit!(model, Xtrain, ytrain)
res = predict(model, Xtest) ; 
@head res.pred 
rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction", ylabel = "Observed").f

####### Discrimination
## The principle is the same as for regression

using Jchemo, JLD2, CairoMakie, JchemoData
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/forages2.jld2")
@load db dat
@names dat
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
## Train ==> Cal + Val 
nval = round(Int, .3 * ntrain)
s = samprand(ntrain, nval; seed = 1234)
Xcal = Xtrain[s.train, :]
ycal = ytrain[s.train]
Xval = Xtrain[s.test, :]
yval = ytrain[s.test]

####---- Plslda
model = plslda()
nlv = 1:30
prior = [:unif]
scal = [false; true]
pars = mpar(prior = prior, scal = scal)
res = gridscore(model, Xcal, ycal, Xval, yval; score = errp, pars, nlv)
typ = string.(res.prior, "-", res.scal)
plotgrid(res.nlv, res.y1, typ; step = 2, xlabel = "Nb. LVs", ylabel = "RMSEP").f
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
model = plslda(nlv = res.nlv[u], prior = res.prior[u])
fit!(model, Xtrain, ytrain)
pred = predict(model, Xtest).pred
@show errp(pred, ytest)
conf(pred, ytest).pct
```
"""
function gridscore(model, Xtrain, Ytrain, X, Y; score::Function, pars::Union{Nothing, NamedTuple} = nothing, 
        nlv::Union{Nothing, Int, AbstractVector{Int}} = nothing, 
        lb::Union{Nothing, Float64, AbstractVector{Float64}} = nothing, 
        verbose::Bool = false)
    ## Multiblock Xbl is allowed
    algo = model.algo
    if isnothing(nlv) && isnothing(lb)
        res = Jchemo.gridscore_br(Xtrain, Ytrain, X, Y; algo, score, pars, verbose)
    elseif !isnothing(nlv)
        res = Jchemo.gridscore_lv(Xtrain, Ytrain, X, Y; algo, score, pars, nlv, verbose)
    elseif !isnothing(lb)
        res = Jchemo.gridscore_lb(Xtrain, Ytrain, X, Y; algo, score, pars, lb, verbose)
    end
    res
end

function gridscore(model::Pipeline, Xtrain, Ytrain, X, Y; score::Function, 
    pars::Union{Nothing, NamedTuple} = nothing, 
    nlv::Union{Nothing, Int, AbstractVector{Int}} = nothing, 
    lb::Union{Nothing, Float64, AbstractVector{Float64}} = nothing, 
    verbose::Bool = false)
    fit!(model, Xtrain, Ytrain)
    K = length(model.model)
    for i = 1:(K - 1)
        Xtrain = transf(model.model[i], Xtrain)
        X = transf(model.model[i], X)
    end
    gridscore(model.model[K], Xtrain, Ytrain, X, Y; score, pars, nlv, lb, verbose)
end

