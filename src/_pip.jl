
struct Pipeline
    mod::Tuple
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

mod1 = model(snv)
npoint = 11 ; deriv = 2 ; degree = 3
mod2 = model(savgol; npoint, deriv, degree)
mod3 = model(plskern; nlv = 15)
mod4 = model(svmr; gamma = 1e3, cost = 100, epsilon = .9)
mod = pip(mod1, mod2, mod3, mod4)
fit!(mod, Xtrain, ytrain)
res = predict(mod, Xtest) ; 
@head res.pred 
rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction",
      ylabel = "Observed").f
```
"""
pip(args...) = Pipeline(values(args))

