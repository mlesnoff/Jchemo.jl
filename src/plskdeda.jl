"""
    plskdeda(X, y, weights = ones(nro(X)); nlv, 
        prior = "unif", h = nothing, a = 1, scal = false)
KDE-LDA on PLS latent variables (PLS-KDE-LDA).
* `X` : X-data.
* `y` : y-data (class membership).
* `weights` : Weights of the observations. 
    Internally normalized to sum to 1. 
* `nlv` : Nb. latent variables (LVs) to compute.
* `prior` : Type of prior probabilities for class membership.
    Posible values are: "unif" (uniform), "prop" (proportional).
* `h` : See `?dmkern`.
* `a` : See `?dmkern`.
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

The principle is the same as functions `plslda` and `plsqda` except 
that densities are estimated from `dmkern` instead of  `dmnorm`.
Function plskdeda` uses function `kdeda`.

## Examples
```julia
using JLD2, StatsBase
using JchemoData
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "iris.jld2") 
@load db dat
pnames(dat)
summ(dat.X)
  
X = dat.X[:, 1:4] 
y = dat.X[:, 5]
n = nro(X)
  
ntrain = 120
s = sample(1:n, ntrain; replace = false) 
Xtrain = X[s, :]
ytrain = y[s]
Xtest = rmrow(X, s)
ytest = rmrow(y, s)

tab(ytrain)
tab(ytest)
aggstat(Matrix(Xtrain), ytrain).X

nlv = 2
fm = plskdeda(Xtrain, ytrain ; nlv = nlv) ;
#fm = plskdeda(Xtrain, ytrain ; nlv = nlv, a = .1) ;
pred = Jchemo.predict(fm, Xtest).pred
tab(pred)
err(pred, ytest)

nlv = 2
fm = plskdeda(Xtrain, ytrain; nlv = nlv) ;
Jchemo.predict(fm, Xtest).posterior
Jchemo.predict(fm, Xtest; nlv = 1).posterior
nlv = 1:2
Jchemo.predict(fm, Xtest; nlv = nlv).posterior[1]
Jchemo.predict(fm, Xtest; nlv = nlv).posterior[2]
```
""" 
function plskdeda(X, y, weights = ones(nro(X)); nlv, 
        prior = "unif", h = nothing, a = 1, scal = false)
    res = dummy(y)
    ni = tab(y).vals
    fm_pls = plskern(X, res.Y, weights; nlv = nlv, scal = scal)
    fm_da = list(nlv)
    @inbounds for i = 1:nlv
        fm_da[i] = kdeda(vcol(fm_pls.T, 1:i), y; prior = prior,
            h = h, a = a)
    end
    fm = (fm_pls = fm_pls, fm_da = fm_da)
    Plslda(fm, res.lev, ni)
end


