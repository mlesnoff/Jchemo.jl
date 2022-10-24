"""
    kplsrda(X, y, weights = ones(size(X, 1)); nlv, kern = "krbf", 
        scal = false, kwargs...)
Discrimination based on kernel partial least squares regression (KPLSR-DA).
* `X` : X-data.
* `y` : Univariate class membership.
* `weights` : Weights of the observations.
* `nlv` : Nb. latent variables (LVs) to compute.
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.
* Other arguments: See `?kplsr`.

This is the same approach as for `plsrda` except that the PLS2 step 
is replaced by a non linear kernel PLS2 (KPLS).

## Examples
```julia
using JLD2
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "forages.jld2") 
@load db dat
pnames(dat)

Xtrain = dat.Xtrain
ytrain = dat.Ytrain.y
Xtest = dat.Xtest
ytest = dat.Ytest.y

tab(ytrain)
tab(ytest)

gamma = .001 
nlv = 15
fm = kplsrda(Xtrain, ytrain; nlv = nlv, gamma = gamma) ;
pnames(fm)
typeof(fm.fm) # = KPLS2 model

res = Jchemo.predict(fm, Xtest) ;
pnames(res)
res.posterior
res.pred
err(res.pred, ytest)

Jchemo.coef(fm.fm)
Jchemo.transform(fm.fm, Xtest)
```
""" 
function kplsrda(X, y, weights = ones(size(X, 1)); nlv, kern = "krbf", 
        scal = false, kwargs...)
    z = dummy(y)
    fm = kplsr(X, z.Y, weights; nlv = nlv, kern = kern, 
        scal = scal, kwargs...)
    Plsrda(fm, z.lev, z.ni)
end

