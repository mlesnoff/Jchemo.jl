"""
    dkplslda(; kwargs...)
    dkplslda(X, y; kwargs...)
    dkplslda(X, y, weights::Weight; kwargs...)
DKPLS-LDA.
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs) to compute. Must be >= 1.
* `kern` : Type of kernel used to compute the Gram matrices. Possible values are: `:krbf`, `:kpol`. See respective 
    functions `krbf` and `kpol` for their keyword arguments.
* `prior` : Type of prior probabilities for class membership. Possible values are: `:prop` (proportionnal), 
    `:unif` (uniform), or a vector (of length equal to the number of classes) giving the prior weight for each class 
    (in case of vector, it must be sorted in the same order as `mlev(y)`).
* Eventual keyword arguments of function `dmkern` (bandwidth definition).
* `scal` : Boolean. If `true`, each column of `X` is scaled by its uncorrected standard deviation.

Same as function `plslda` (PLS-LDA) except that a direct kernel PLSR (function `dkplsr`), instead of a PLSR 
(function `plskern`), is run on the Y-dummy table. 

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
model = dkplslda(; nlv, gamma) 
#model = dkplslda(; nlv, gamma, prior = :unif) 
#model = dkplsqda(; nlv, gamma, alpha = .5) 
#model = dkplskdeda(; nlv, gamma, a = .5) 
fit!(model, Xtrain, ytrain)
@names model
@names fitm = model.fitm

fitm.lev
fitm.ni

fitm_emb = fitm.fitm_emb ;
typeof(fitm_emb)
@names fitm_emb 
typeof(fitm_emb.fitm)

@head transf(model, Xtrain)
@head fitm_emb.fitm.T

@head transf(model, Xtest)
@head transf(model, Xtest; nlv = 3)

res = predict(model, Xtest) ;
@names res
@head res.posterior
@head res.pred
errp(res.pred, ytest)
conf(res.pred, ytest).cnt

predict(model, Xtest; nlv = 1:2).pred
```
""" 
dkplslda(; kwargs...) = JchemoModel(dkplslda, nothing, kwargs)

function dkplslda(X, y; kwargs...)
    par = recovkw(ParKplsda, kwargs).par
    Q = eltype(X[1, 1])
    weights = mweightcla(Q, y; prior = par.prior)
    dkplslda(X, y, weights; kwargs...)
end

function dkplslda(X, y, weights::Weight; kwargs...)
    par = recovkw(ParKplsda, kwargs).par
    @assert par.nlv >= 1 "Argument 'nlv' must be in >= 1"   
    res = dummy(y)
    ni = tab(y).vals
    fitm_emb = dkplsr(X, res.Y, weights; kwargs...)
    fitm_da = list(Lda, par.nlv)
    @inbounds for i = 1:par.nlv
        fitm_da[i] = lda(vcol(fitm_emb.fitm.T, 1:i), y, weights; kwargs...)
    end
    Plsprobda(fitm_emb, fitm_da, res.lev, ni, par) 
end




