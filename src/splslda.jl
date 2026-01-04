"""
    splslda(; kwargs...)
    splslda(X, y; kwargs...)
    splslda(X, y, weights::Weight; kwargs...)
Sparse PLS-LDA.
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

Same as function `plslda` (PLSR-LDA) except that a sparse PLSR (function `splsr`), instead of a PLSR, 
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
model = splslda(; nlv, meth, nvar) 
#model = splsqda(; nlv, meth, nvar, alpha = .1) 
#model = splskdeda(; nlv, meth, nvar) 
#model = splskdeda(; nlv, meth, nvar, a = .8) 
fit!(model, Xtrain, ytrain)
@names model
@names fitm = model.fitm

fitm.lev
fitm.ni
fitm.priors

fitm_emb = fitm.fitm_emb ;
typeof(fitm_emb)
@names fitm_emb  
fitm_emb.sellv
fitm_emb.sel
@head fitm_emb.T
@head transf(model, Xtrain)
@head transf(model, Xtest)
@head transf(model, Xtest; nlv = 3)

coef(fitm_emb)

res = predict(model, Xtest) ;
@names res
@head res.posterior
@head res.pred
errp(res.pred, ytest)
conf(res.pred, ytest).cnt

predict(model, Xtest; nlv = 1:2).pred
summary(fitm_emb, Xtrain)
```
""" 
splslda(; kwargs...) = JchemoModel(splslda, nothing, kwargs)

function splslda(X, y; kwargs...)
    par = recovkw(ParSplsda, kwargs).par
    Q = eltype(X[1, 1])
    weights = mweightcla(Q, y; prior = par.prior)
    splslda(X, y, weights; kwargs...)
end

function splslda(X, y, weights::Weight; kwargs...)
    par = recovkw(ParSplsda, kwargs).par
    @assert par.nlv >= 1 "Argument 'nlv' must be in >= 1"   
    res = dummy(y)
    ni = tab(y).vals
    priors = aggsumv(weights.w, y).val  # output not used, only for information
    fitm_emb = splsr(X, res.Y, weights; kwargs...)
    fitm_da = list(Lda, par.nlv)
    @inbounds for i = 1:par.nlv
        fitm_da[i] = lda(vcol(fitm_emb.T, 1:i), y, weights; kwargs...)
    end
    Plsprobda(fitm_emb, fitm_da, ni, priors, res.lev, par)  
end






