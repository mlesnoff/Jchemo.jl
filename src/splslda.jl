"""
    splslda(; kwargs...)
    splslda(X, y; kwargs...)
    splslda(X, y, weights::Weight; kwargs...)
Sparse PLS-LDA.
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`). 
Keyword arguments: 
* `nlv` : Nb. latent variables (LVs) to compute.
    Must be >= 1.
* `msparse` : Method used for the sparse thresholding. 
    Possible values are: `:soft`, `:mix`, 
    `:hard`. See thereafter.
* `delta` : Only used if `msparse = `:soft`. Range for the 
    thresholding on the loadings (after they are standardized 
    to their maximal absolute value). Must ∈ [0, 1].
    Higher is `delta`, stronger is the thresholding. 
* `nvar` : Only used if `msparse` = `:mix` or `:hard`.
    Nb. variables (`X`-columns) selected for each principal
    component (PC). Can be a single integer (i.e. same nb. 
    of variables for each PC), or a vector of length `nlv`.   
* `prior` : Type of prior probabilities for class 
    membership. Possible values are: `:unif` (uniform), 
    `:prop` (proportional).
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

Same as function `plslda` (PLS-LDA) except that 
a sparse PLSR (function `splskern`), instead of a 
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
msparse = :mix ; nvar = 10
mo = splslda(; nlv, 
    msparse, nvar) 
fit!(mo, Xtrain, ytrain)
pnames(mo)
pnames(mo.fm)
fm = mo.fm ;
fm.lev
fm.ni

fmpls = fm.fm.fmpls ; 
@head fmpls.T
@head transf(mo, Xtrain)
@head transf(mo, Xtest)
@head transf(mo, Xtest; nlv = 3)

coef(fmpls)

res = predict(mo, Xtest) ;
pnames(res)
@head res.posterior
@head res.pred
errp(res.pred, ytest)
confusion(res.pred, ytest).cnt

predict(mo, Xtest; nlv = 1:2).pred
summary(fm.fm, Xtrain)
```
""" 
function splslda(X, y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    splslda(X, y, weights; kwargs...)
end

function splslda(X, y, weights::Weight; kwargs...)
    par = recovkwargs(Par, kwargs)
    @assert par.nlv >= 1 "Argument 'nlv' must be in >= 1"   
    res = dummy(y)
    ni = tab(y).vals
    fmpls = splskern(X, res.Y, weights; kwargs...)
    fmda = list(Lda, par.nlv)
    @inbounds for i = 1:par.nlv
        fmda[i] = lda(fmpls.T[:, 1:i], y, weights; 
            kwargs...)
    end
    fm = (fmpls = fmpls, fmda = fmda)
    Plslda(fm, res.lev, ni)
end






