"""
    gridscore(model::Pipeline, Xtrain, Ytrain, X, Y; score, pars = nothing, nlv = nothing, lb = nothing, 
        verbose = false) 
`gridscore` for a pipeline model.

In the present version of the function, only the last model of the pipeline (= the final predictor) 
is tuned. Therefore, argument `pars` must only contain parameters for this last model.

For other details, see usual function `gridscore`. 

## Examples
```julia
using Jchemo, JLD2, CairoMakie, JchemoData
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "cassav.jld2") 
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
ntrain = nro(Xtrain)
ntest = nro(Xtest)
ntot = ntrain + ntest
(ntot = ntot, ntrain, ntest)
## Building Cal and Val 
## within Train
nval = round(Int, .3 * ntrain)
s = samprand(ntrain, nval)
Xcal = Xtrain[s.train, :]
ycal = ytrain[s.train]
Xval = Xtrain[s.test, :]
yval = ytrain[s.test]

####-- Pipeline Snv :> Savgol :> Plsr   (Only the last model is tuned)
## model1
model1 = snv()
## model2 
npoint = 11 ; deriv = 2 ; degree = 3
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
nlv = 15 ; scal = true
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
```
"""
function gridscore(model::Pipeline, Xtrain, Ytrain, X, Y; score, pars = nothing, nlv = nothing, lb = nothing, 
        verbose = false)
    fit!(model, Xtrain, Ytrain)
    K = length(model.model)
    for i = 1:(K - 1)
        Xtrain = transf(model.model[i], Xtrain)
        X = transf(model.model[i], X)
    end
    gridscore(model.model[K], Xtrain, Ytrain, X, Y; score, pars, nlv, lb, verbose)
end

