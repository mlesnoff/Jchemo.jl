"""
    splsrda(; kwargs...)
    splsrda(X, y; kwargs...)
    splsrda(X, y, weights::ProbabilityWeights; kwargs...)
Sparse PLSR-DA.
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`). 
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
* `scal` : Symbol defining the scaling. Possible values are: `std`, `prt` (pareto) and `mad`..    

Same as function `plsrda` (PLSR-DA) except that a sparse PLSR (function `splsr`), instead of a PLSR, 
is run on the Y-dummy table. 

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
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
typeof(fitm.fitm_emb) 
@names fitm.fitm_emb

fitm.lev
fitm.ni
fitm.priors

fitm.fitm_emb.sellv
fitm.fitm_emb.sel

@head fitm.fitm_emb.T
@head transf(model, Xtrain)
@head transf(model, Xtest)
@head transf(model, Xtest, 3)

coef(fitm.fitm_emb)

res = predict(model, Xtest) ;
@names res
@head res.posterior
@head res.pred
errp(res.pred, ytest)
conf(res.pred, ytest).cnt

predict(model, Xtest, 1:2).pred
summary(fitm.fitm_emb, Xtrain)
```
""" 
splsrda(; kwargs...) = JchemoModel(splsrda, nothing, kwargs)

function splsrda(X, y; kwargs...)
    par = recovkw(ParSplsda{Q}, kwargs).par
    Q = eltype(X[1, 1])
    weights = pweightcla(Q, y; prior = par.prior)
    splsrda(X, y, weights; kwargs...)
end

function splsrda(X, y, weights::ProbabilityWeights; kwargs...)
    par = recovkw(ParSplsda{Q}, kwargs).par
    res = dummy(Q, y)
    ni = tab(y).vals
    priors = aggsumv(weights.values, vec(y)).val  # output not used, only for information
    fitm_emb = splsr(X, res.Y, weights; kwargs...)
    par.nlv = fitm_emb.par.nlv
    Plsrda(fitm_emb, ni, priors, res.lev, par)
end


