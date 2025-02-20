"""
    plslda(; kwargs...)
    plslda(X, y; kwargs...)
    plslda(X, y, weights::Weight; kwargs...)
LDA on PLS latent variables (PLS-LDA).
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
* `scal` : Boolean. If `true`, each column of `X` 
    and Ydummy is scaled by its uncorrected standard deviation
    in the PLS computation.

LDA on PLS latent variables. The method is as follows:

1) The training variable `y` (univariate class membership) is 
    transformed to a dummy table (Ydummy) containing nlev columns, 
    where nlev is the number of classes present in `y`. Each column of 
    Ydummy is a dummy (0/1) variable. 
2) A multivariate PLSR (PLSR2) is run on {`X`, Ydummy}, returning 
    a score matrix `T`.
3) A LDA is done on {`T`, `y`}, returning estimates of posterior probabilities
    (∊ [0, 1]) of class membership.
4) For a given observation, the final prediction is the class 
    corresponding to the dummy variable for which the probability 
    estimate is the highest.

In the high-level version of the present functions, the observation 
weights are automatically defined by the given priors (argument `prior`): 
the sub-totals by class of the observation weights are set equal to the prior 
probabilities. The low-level version (argument `weights`) allows to implement 
other choices.

## Examples
```julia
using Jchemo, JchemoData, JLD2
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
model = plslda(; nlv) 
#model = plslda(; nlv, prior = :prop) 
#model = plsqda(; nlv, alpha = .1) 
fit!(model, Xtrain, ytrain)
@names model
@names model.fitm
fitm = model.fitm ;
fitm.lev
fitm.ni

embfitm = fitm.fitm.embfitm ;
@head embfitm.T
@head transf(model, Xtrain)
@head transf(model, Xtest)
@head transf(model, Xtest; nlv = 3)

coef(embfitm)

res = predict(model, Xtest) ;
@names res
@head res.posterior
@head res.pred
errp(res.pred, ytest)
conf(res.pred, ytest).cnt

predict(model, Xtest; nlv = 1:2).pred
summary(embfitm, Xtrain)
```
""" 
plslda(; kwargs...) = JchemoModel(plslda, nothing, kwargs)

function plslda(X, y; kwargs...)
    par = recovkw(ParPlsda, kwargs).par
    Q = eltype(X[1, 1])
    weights = mweightcla(Q, y; prior = par.prior)
    plslda(X, y, weights; kwargs...)
end

function plslda(X, y, weights::Weight; kwargs...)
    par = recovkw(ParPlsda, kwargs).par
    @assert par.nlv >= 1 "Argument 'nlv' must be in >= 1"   
    res = dummy(y)
    ni = tab(y).vals
    embfitm = plskern(X, res.Y, weights; kwargs...)
    dafitm = list(Lda, par.nlv)
    @inbounds for i = 1:par.nlv
        dafitm[i] = lda(vcol(embfitm.T, 1:i), y, weights; kwargs...)
    end
    fitm = (embfitm = embfitm, dafitm = dafitm)
    Plsprobda(fitm, res.lev, ni, par)
end

""" 
    transf(object::Plsprobda, X; nlv = nothing)
Compute latent variables (LVs = scores) from 
    a fitted model.
* `object` : The fitted model.
* `X` : Matrix (m, p) for which LVs are computed.
* `nlv` : Nb. LVs to consider.
""" 
function transf(object::Plsprobda, X; nlv = nothing)
    transf(object.fitm.embfitm, X; nlv)
end

"""
    predict(object::Plsprobda, X; nlv = nothing)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
* `nlv` : Nb. LVs, or collection of nb. LVs, to consider. 
""" 
function predict(object::Plsprobda, X; nlv = nothing)
    X = ensure_mat(X)
    Q = eltype(X)
    Qy = eltype(object.lev)
    m = nro(X)
    a = size(object.fitm.embfitm.T, 2)
    isnothing(nlv) ? nlv = a : nlv = (max(minimum(nlv), 1):min(maximum(nlv), a))
    le_nlv = length(nlv)
    pred = list(Matrix{Qy}, le_nlv)
    posterior = list(Matrix{Q}, le_nlv)
    T = transf(object.fitm.embfitm, X)
    @inbounds for i in eachindex(nlv)
        znlv = nlv[i]
        zres = predict(object.fitm.dafitm[znlv], vcol(T, 1:znlv))
        z =  mapslices(argmax, zres.posterior; dims = 2) 
        pred[i] = reshape(recod_indbylev(z, object.lev), m, 1)
        posterior[i] = zres.posterior
    end 
    if le_nlv == 1
        pred = pred[1]
        posterior = posterior[1]
    end
    (pred = pred, posterior = posterior)
end


