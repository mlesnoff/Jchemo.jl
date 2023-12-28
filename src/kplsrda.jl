"""
    kplsrda(X, y, weights = ones(nro(X)); nlv, kern = :krbf, 
        scal::Bool = false, kwargs...)
Discrimination based on kernel partial least squares regression (KPLSR-DA).
* `X` : X-data.
* `y` : Univariate class membership.
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`). 
* `nlv` : Nb. latent variables (LVs) to compute.
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.
* Other arguments to pass in the kernel: See function `kplsr`.

This is the same approach as for `plsrda` except that the PLS2 step 
is replaced by a non linear kernel PLS2 (KPLS).

## Examples
```julia
using JLD2
using JchemoData
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
confusion(res.pred, ytest).cnt

Jchemo.coef(fm.fm)
transf(fm.fm, Xtest)
```
""" 
function kplsrda(X, y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    kplsrda(X, y, weights; kwargs...)
end

function kplsrda(X, y, weights::Weight; 
        kwargs...)
    res = dummy(y)
    ni = tab(y).vals
    fm = kplsr(X, res.Y, weights; 
        kwargs...)
    Plsrda(fm, res.lev, ni)
end

