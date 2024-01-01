"""
    plskdeda(X, y, weights = ones(nro(X)); nlv, 
        prior = :unif, h = nothing, a = 1, scal::Bool = false)
KDE-LDA on PLS latent variables (PLS-KDE-LDA).
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
* `weights` : Weights of the observations. 
    Internally normalized to sum to 1. 
* `nlv` : Nb. latent variables (LVs) to compute.
* `prior` : Type of prior probabilities for class membership.
    Possible values are: :unif (uniform; default), :prop (proportional).
* `h` : See function `dmkern`.
* `a` : See function `dmkern`.
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

The principle is the same as functions `plslda` and `plsqda` except 
that densities are estimated from `dmkern` instead of  `dmnorm`.

See examples in `?plslda` for detailed outputs.

## Examples
```julia
using JLD2
using JchemoData
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

nlv = 20
fm = plskdeda(Xtrain, ytrain; nlv) ;
#fm = plskdeda(Xtrain, ytrain; nlv = nlv, a = .5) ;
res = Jchemo.predict(fm, Xtest) ;
pred = res.pred
errp(pred, ytest)
confusion(pred, ytest).cnt
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


