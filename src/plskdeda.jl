"""
    plskdeda(; kwargs...)
    plskdeda(X, y; kwargs...)
    plskdeda(X, y, weights::Weight; kwargs...)
KDE-DA on PLS latent variables (PLS-KDEDA).
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs) to compute.
    Must be >= 1.
* `prior` : Type of prior probabilities for class 
    membership. Possible values are: `:unif` (uniform), 
    `:prop` (proportional), or a vector (of length equal to 
    the number of classes) giving the prior weight for each class 
    (in case of vector, it must be sorted in the same order as `mlev(y)`).
* Keyword arguments of function `dmkern` (bandwidth 
    definition) can also be specified here.
* `scal` : Boolean. If `true`, each column of `X` 
    and Ydummy is scaled by its uncorrected standard deviation
    in the PLS computation.

The principle is the same as function `plsqda` except that the 
densities by class are estimated from `dmkern` instead of `dmnorm`.

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
model = plskdeda(; nlv) 
#model = plskdeda(; nlv, a = .5)
fit!(model, Xtrain, ytrain)
pnames(model)
pnames(model.fitm)
fitm = model.fitm ;
fitm.lev
fitm.ni

fitm_emb = fitm.fitm.fitm_emb ;
@head fitm_emb.T
@head transf(model, Xtrain)
@head transf(model, Xtest)
@head transf(model, Xtest; nlv = 3)

coef(fitm_emb)

res = predict(model, Xtest) ;
pnames(res)
@head res.posterior
@head res.pred
errp(res.pred, ytest)
conf(res.pred, ytest).cnt

predict(model, Xtest; nlv = 1:2).pred
summary(fitm_emb, Xtrain)
```
""" 
plskdeda(; kwargs...) = JchemoModel(plskdeda, nothing, kwargs)

function plskdeda(X, y; kwargs...)
    par = recovkw(ParPlskdeda, kwargs).par
    Q = eltype(X[1, 1])
    weights = mweightcla(Q, y; prior = par.prior)
    plskdeda(X, y, weights; kwargs...)
end

function plskdeda(X, y, weights::Weight; kwargs...)
    par = recovkw(ParPlskdeda, kwargs).par
    @assert par.nlv >= 1 "Argument 'nlv' must be in >= 1"   
    res = dummy(y)
    ni = tab(y).vals
    fitm_emb = plskern(X, res.Y, weights; kwargs...)
    fitm_da = list(Kdeda, par.nlv)
    @inbounds for i = 1:par.nlv
        fitm_da[i] = kdeda(vcol(fitm_emb.T, 1:i), y; kwargs...)
    end
    fitm = (fitm_emb = fitm_emb, fitm_da = fitm_da)
    Plsprobda(fitm, res.lev, ni, par)
end


