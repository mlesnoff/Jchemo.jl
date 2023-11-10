"""
    plskdeda(X, y, weights = ones(nro(X)); nlv, 
        prior = :unif, h = nothing, a = 1, scal::Bool = false)
KDE-LDA on PLS latent variables (PLS-KDE-LDA).
* `X` : X-data.
* `y` : y-data (class membership).
* `weights` : Weights of the observations. 
    Internally normalized to sum to 1. 
* `nlv` : Nb. latent variables (LVs) to compute.
* `prior` : Type of prior probabilities for class membership.
    Possible values are: :unif (uniform; default), :prop (proportional).
* `h` : See `?dmkern`.
* `a` : See `?dmkern`.
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
fm = plskdeda(Xtrain, ytrain; nlv = nlv) ;
#fm = plskdeda(Xtrain, ytrain; nlv = nlv, a = .5) ;
res = Jchemo.predict(fm, Xtest) ;
pred = res.pred
err(pred, ytest)
confusion(pred, ytest).cnt
```
""" 
function plskdeda(X, y, weights = ones(nro(X)); nlv, 
        prior = :unif, h = nothing, a = 1, scal::Bool = false)
    res = dummy(y)
    ni = tab(y).vals
    fm_pls = plskern(X, res.Y, weights; nlv = nlv, scal = scal)
    fm_da = list(nlv)
    @inbounds for i = 1:nlv
        fm_da[i] = kdeda(vcol(fm_pls.T, 1:i), y; prior = prior,
            h = h, a = a)
    end
    fm = (fm_pls = fm_pls, fm_da = fm_da)
    Plslda(fm, res.lev, ni)
end


