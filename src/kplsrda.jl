"""
    kplsrda(; kwargs...)
    kplsrda(X, y; kwargs...)
    kplsrda(X, y, weights::Weight; kwargs...)
Discrimination based on kernel partial least squares
    regression (KPLSR-DA).
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`). 
Keyword arguments: 
* `nlv` : Nb. latent variables (LVs) to compute.
* `kern` : Type of kernel used to compute the Gram matrices.
    Possible values are: `:krbf`, `:kpol`. See respective 
    functions `krbf` and `kpol` for their keyword arguments.
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

Same as function `plsrda` (PLSR-DA) except that 
a kernel PLSR (function `kplsr`), instead of a 
PLSR (function `plskern`), is run on the Y-dummy table. 

## Examples
```julia
using JchemoData, JLD2
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/forages2.jld2")
@load db dat
pnames(dat)
X = dat.X
Y = dat.Y
n = nro(X) 
s = Bool.(Y.test)
Xtrain = rmrow(X, s)
ytrain = rmrow(Y.typ, s)
Xtest = X[s, :]
ytest = Y.typ[s]
ntrain = nro(Xtrain)
ntest = nro(Xtest)
(ntot = n, ntrain, ntest)
tab(ytrain)
tab(ytest)

nlv = 15
kern = :krbf ; gamma = .001 
scal = true
mo = kplsrda(; nlv,
    kern, gamma, scal) 
fit!(mo, Xtrain, ytrain)
pnames(mo)
pnames(mo.fm)
fm = mo.fm ;
fm.lev
fm.ni

@head fm.fm.T
@head transf(mo, Xtrain)
@head transf(mo, Xtest)
@head transf(mo, Xtest; nlv = 3)

coef(fm.fm)

res = predict(mo, Xtest) ;
pnames(res)
@head res.posterior
@head res.pred
errp(res.pred, ytest)
confusion(res.pred, ytest).cnt

predict(mo, Xtest; nlv = 1:2).pred
```
""" 
function kplsrda(X, y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    kplsrda(X, y, weights; kwargs...)
end

function kplsrda(X, y, weights::Weight; kwargs...)
    res = dummy(y)
    ni = tab(y).vals
    fm = kplsr(X, res.Y, weights; kwargs...)
    Plsrda(fm, res.lev, ni)
end

