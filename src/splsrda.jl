"""
    splsrda(; kwargs...)
    splsrda(X, y; kwargs...)
    splsrda(X, y, weights::Weight; kwargs...)
Sparse PLSR-DA.
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g. function `mweight`). 
Keyword arguments: 
* `nlv` : Nb. latent variables (LVs) to compute.
* `meth` : Method used for the sparse thresholding. Possible values are: `:soft`, `:hard`. See thereafter.
* `nvar` : Nb. variables (`X`-columns) selected for each LV. Can be a single integer (i.e. same nb. 
    of variables for each LV), or a vector of length `nlv`.   
* `prior` : Type of prior probabilities for class membership. Possible values are: `:prop` (proportionnal), 
    `:unif` (uniform), or a vector (of length equal to the number of classes) giving the prior weight for each class 
    (in case of vector, it must be sorted in the same order as `mlev(y)`).
* `tol` : Only when q > 1; tolerance used in function `snipals_shen`. 
* `maxit` : Only when q > 1; maximum nb. of iterations used in function `snipals_shen`.    
* `scal` : Boolean. If `true`, each column of `X` and `Y` is scaled by its uncorrected standard deviation.    

Same as function `plsrda` (PLSR-DA) except that a sparse PLSR (function `splsr`), instead of a PLSR, 
is run on the Y-dummy table. 

## Examples
```julia
using JchemoData, JLD2
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/forages2.jld2")
@load db dat
@names dat
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
#meth = :soft ; nvar = 1
model = splsrda(; nlv, meth, nvar) 
fit!(model, Xtrain, ytrain)
@names model
fitm = model.fitm ;
typeof(fitm)
@names fitm
typeof(fitm.fitm) 
@names fitm.fitm
fitm.lev
fitm.ni

fitm.fitm.sellv
fitm.fitm.sel

@head fitm.fitm.T
@head transf(model, Xtrain)
@head transf(model, Xtest)
@head transf(model, Xtest; nlv = 3)

coef(fitm.fitm)

res = predict(model, Xtest) ;
@names res
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
    priors = aggsumv(weights.w, y).val  # output not used, only for information
    fitm = splsr(X, res.Y, weights; kwargs...)
    Plsrda(fitm, priors, ni, res.lev, par)
end


