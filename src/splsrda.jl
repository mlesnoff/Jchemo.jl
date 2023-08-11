"""
    splsrda(X, y, weights = ones(nro(X)); nlv,
        scal::Bool = false)
Sparse PLSR-DA.
* `X` : X-data.
* `y` : y-data (class membership).
* `weights` : Weights of the observations. Internally normalized to sum to 1. 
* `nlv` : Nb. latent variables (LVs) to compute.
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

Same as function `plsrda` (PLSR-DA) except that function `splskern` is used 
instead function `plskern`. See the help of the respectve functions. 

## Examples
```julia
using JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/forages2.jld2") 
@load db dat
pnames(dat)

X = dat.X 
Y = dat.Y 
s = Bool.(Y.test)
Xtrain = rmrow(X, s)
ytrain = rmrow(Y.typ, s)
Xtest = X[s, :]
ytest = Y.typ[s]

tab(ytrain)
tab(ytest)

nlv = 15
fm = splsrda(Xtrain, ytrain; nlv = nlv) ;
pnames(fm)
typeof(fm.fm) # = PLS2 model

res = Jchemo.predict(fm, Xtest) ;
pnames(res)
res.posterior
res.pred
err(res.pred, ytest)
confusion(res.pred, ytest).cnt

Jchemo.transform(fm, Xtest)

Jchemo.transform(fm.fm, Xtest)
Jchemo.coef(fm.fm)
summary(fm.fm, Xtrain)

Jchemo.predict(fm, Xtest; nlv = 1:2).pred
```
""" 
function splsrda(X, y, weights = ones(nro(X)); nlv,
        scal::Bool = false)
    res = dummy(y)
    ni = tab(y).vals
    fm = splskern(X, res.Y, weights; nlv = nlv, scal = scal)
    Plsrda(fm, res.lev, ni)
end


