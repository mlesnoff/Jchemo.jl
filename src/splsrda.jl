"""
    splsrda(; kwargs...)
    splsrda(X, y; kwargs...)
    splsrda(X, y, weights::Weight; kwargs...)
Sparse PLSR-DA.
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`). 
Keyword arguments: 
* `nlv` : Nb. latent variables (LVs) to compute.
* `meth` : Method used for the sparse thresholding. 
    Possible values are: `:soft`, `:softs`, 
    `:hard`. See thereafter.
* `delta` : Only used if `meth = :softs`. Constant used in function 
   `soft` for the thresholding on the loadings (after they are 
    standardized to their maximal absolute value). Must ∈ [0, 1].
    Higher is `delta`, stronger is the thresholding. 
* `nvar` : Only used if `meth = :soft` or `meth = :hard`.
    Nb. variables (`X`-columns) selected for each principal
    component (PC). Can be a single integer (i.e. same nb. 
    of variables for each PC), or a vector of length `nlv`.   
* `prior` : Type of prior probabilities for class 
    membership. Possible values are: `:unif` (uniform), 
    `:prop` (proportional), or a vector (of length equal to 
    the number of classes) giving the prior weight for each class 
    (in case of vector, it must be sorted in the same order as `mlev(y)`).
* `scal` : Boolean. If `true`, each column of `X` 
    and Ydummy is scaled by its uncorrected standard deviation.

Same as function `plsrda` (PLSR-DA) except that 
a sparse PLSR (function `splsr`), instead of a 
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
meth = :soft ; nvar = 10
model = splsrda(; nlv, meth, nvar) 
fit!(model, Xtrain, ytrain)
pnames(model)
pnames(model.fitm)
fitm = model.fitm ;
fitm.lev
fitm.ni

@head fitm.fitm.T
@head transf(model, Xtrain)
@head transf(model, Xtest)
@head transf(model, Xtest; nlv = 3)

coef(fitm.fitm)

res = predict(model, Xtest) ;
pnames(res)
@head res.posterior
@head res.pred
errp(res.pred, ytest)
conf(res.pred, ytest).cnt

predict(model, Xtest; nlv = 1:2).pred
summary(fitm.fitm, Xtrain)
```
""" 
splsrda(; kwargs...) = JchemoModel(splsrda, nothing, kwargs)

function splsrda(X, y; kwargs...)
    par = recovkw(ParSplsda, kwargs).par
    Q = eltype(X[1, 1])
    weights = mweightcla(Q, y; prior = par.prior)
    splsrda(X, y, weights; kwargs...)
end

function splsrda(X, y, weights::Weight; kwargs...)
    par = recovkw(ParSplsda, kwargs).par
    res = dummy(y)
    ni = tab(y).vals
    fitm = splsr(X, res.Y, weights; kwargs...)
    Plsrda(fitm, res.lev, ni, par)
end


