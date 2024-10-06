"""
    dkplsrda(X, y; kwargs...)
    dkplsrda(X, y, weights::Weight; kwargs...)
Discrimination based on direct kernel partial least squares 
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
* `prior` : Type of prior probabilities for class 
    membership. Possible values are: `:unif` (uniform), 
    `:prop` (proportional), or a vector (of length equal to 
    the number of classes) giving the prior weight for each class 
    (in case of vector, it must be sorted in the same order as `mlev(y)`).
* `scal` : Boolean. If `true`, each column of `X` 
    and Ydummy is scaled by its uncorrected standard deviation.

Same as function `plsrda` (PLSR-DA) except that 
a direct kernel PLSR (function `dkplsr`), instead of a 
PLSR (function `plskern`), is run on the Y-dummy table. 

## Examples
```julia
using Jchemo, JchemoData, JLD2
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
model = mod_(dkplsrda; nlv, kern, gamma, scal) 
fit!(model, Xtrain, ytrain)
pnames(model)
pnames(model.fm)
fm = model.fm ;
fm.lev
fm.ni

@head fm.fm.T
@head transf(model, Xtrain)
@head transf(model, Xtest)
@head transf(model, Xtest; nlv = 3)

coef(fm.fm)

res = predict(model, Xtest) ;
pnames(res)
@head res.posterior
@head res.pred
errp(res.pred, ytest)
conf(res.pred, ytest).cnt

predict(model, Xtest; nlv = 1:2).pred
```
""" 
function dkplsrda(X, y; kwargs...)
    par = recovkw(ParKplsda, kwargs).par
    Q = eltype(X[1, 1])
    weights = mweightcla(Q, y; prior = par.prior)
    dkplsrda(X, y, weights; kwargs...)
end

function dkplsrda(X, y, weights::Weight; kwargs...)
    par = recovkw(ParKplsda, kwargs).par
    res = dummy(y)
    ni = tab(y).vals
    fm = dkplsr(X, res.Y, weights; kwargs...)
    Plsrda(fm, res.lev, ni, par)
end

