
struct Pipeline
    model::Tuple
end

"""
    pip(args...)
Build a pipeline of models.
* `args...` : Succesive models, see examples.

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

## Pipeline Snv :> Savgol :> Pls :> Svmr

model1 = mod_(snv)
npoint = 11 ; deriv = 2 ; degree = 3
model2 = mod_(savgol; npoint, deriv, degree)
model3 = mod_(plskern; nlv = 15)
mod4 = mod_(svmr; gamma = 1e3, cost = 100, epsilon = .9)
model = pip(model1, model2, model3, mod4)
fit!(model, Xtrain, ytrain)
res = predict(model, Xtest) ; 
@head res.pred 
rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction",
      ylabel = "Observed").f
```
"""
pip(args...) = Pipeline(values(args))

