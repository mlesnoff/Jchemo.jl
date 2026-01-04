"""
    kplslda(; kwargs...)
    kplslda(X, y; kwargs...)
    kplslda(X, y, weights::Weight; kwargs...)
KPLS-LDA.
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs) to compute. Must be >= 1
* `kern` : Type of kernel used to compute the Gram matrices. Possible values are: `:krbf`, `:kpol`. See respective 
    functions `krbf` and `kpol` for their keyword arguments.
* `prior` : Type of prior probabilities for class membership. Possible values are: `:prop` (proportionnal), 
    `:unif` (uniform), or a vector (of length equal to the number of classes) giving the prior weight for each class 
    (in case of vector, it must be sorted in the same order as `mlev(y)`).
* `scal` : Boolean. If `true`, each column of `X` and Ydummy is scaled by its uncorrected standard deviation
    in the PLS computation.

Same as function `plslda` (PLS-LDA) except that a kernel PLSR (function `kplsr`), instead of a PLSR (function `plskern`), 
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
gamma = .1
model = kplslda(; nlv, gamma) 
#model = kplslda(; nlv, gamma, prior = :unif) 
#model = kplsqda(; nlv, gamma, alpha = .5) 
#model = kplskdeda(; nlv, gamma, a = .5) 
fit!(model, Xtrain, ytrain)
@names model
fitm = model.fitm ;
typeof(fitm)
@names fitm

fitm.lev
fitm.ni

fitm_emb = fitm.fitm_emb ;
typeof(fitm_emb)
@names fitm_emb 
@head transf(model, Xtrain)
@head fitm_emb.T

@head transf(model, Xtest)
@head transf(model, Xtest; nlv = 3)

fitm_da = fitm.fitm_da ;
typeof(fitm_da)

res = predict(model, Xtest) ;
@names res
@head res.posterior
@head res.pred
errp(res.pred, ytest)
conf(res.pred, ytest).cnt

predict(model, Xtest; nlv = 1:2).pred
```
""" 
kplslda(; kwargs...) = JchemoModel(kplslda, nothing, kwargs)

function kplslda(X, y; kwargs...)
    par = recovkw(ParKplsda, kwargs).par
    Q = eltype(X[1, 1])
    weights = mweightcla(Q, y; prior = par.prior)
    kplslda(X, y, weights; kwargs...)
end

function kplslda(X, y, weights::Weight; kwargs...)
    par = recovkw(ParKplsda, kwargs).par
    @assert par.nlv >= 1 "Argument 'nlv' must be in >= 1"   
    res = dummy(y)
    ni = tab(y).vals
    fitm_emb = kplsr(X, res.Y, weights; kwargs...)
    fitm_da = list(Lda, par.nlv)
    @inbounds for a = 1:par.nlv
        fitm_da[a] = lda(vcol(fitm_emb.T, 1:a), y, weights; kwargs...)
    end
    Plsprobda(fitm_emb, fitm_da, ni, res.lev, par) 
end



