"""
    plskdeda(; kwargs...)
    plskdeda(X, y; kwargs...)
    plskdeda(X, y, weights::Weight; 
        kwargs...)
KDE-LDA on PLS latent variables (PLS-KDE-LDA).
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs) to compute.
    Must be >= 1.
* `prior` : Type of prior probabilities for class 
    membership. Possible values are: `:unif` (uniform), 
    `:prop` (proportional).
* Keyword arguments of function `dmkern` (bandwidth 
    definition) can also be specified here.
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

The principle is the same as functions `plslda` and 
`plslda` except that densities are estimated from `dmkern` 
instead of `dmnorm`.

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
mod = plskdeda(; nlv) 
#mod = plskdeda(; nlv, a_kde = .5)
fit!(mod, Xtrain, ytrain)
pnames(mod)
pnames(mod.fm)
fm = mod.fm ;
fm.lev
fm.ni

fmpls = fm.fm.fmpls ;
@head fmpls.T
@head transf(mod, Xtrain)
@head transf(mod, Xtest)
@head transf(mod, Xtest; nlv = 3)

coef(fmpls)

res = predict(mod, Xtest) ;
pnames(res)
@head res.posterior
@head res.pred
errp(res.pred, ytest)
confusion(res.pred, ytest).cnt

predict(mod, Xtest; nlv = 1:2).pred
summary(fmpls, Xtrain)
```
""" 
function plskdeda(X, y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    plskdeda(X, y, weights; 
        kwargs...)
end

function plskdeda(X, y, weights::Weight; 
        kwargs...)
    par = recovkwargs(Par, kwargs)
    @assert par.nlv >= 1 "Argument 'nlv' must be in >= 1"   
    res = dummy(y)
    ni = tab(y).vals
    fmpls = plskern(X, res.Y, weights; 
        kwargs...)
    fmda = list(Kdeda, par.nlv)
    @inbounds for i = 1:par.nlv
        fmda[i] = kdeda(vcol(fmpls.T, 1:i), y; 
            kwargs...)
    end
    fm = (fmpls = fmpls, fmda = fmda)
    Plslda(fm, res.lev, ni)
end


