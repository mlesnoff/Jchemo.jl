"""
    plslda(; kwargs...)
    plslda(X, y; kwargs...)
    plslda(X::Matrix{Q}, y::Vector{String}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: Float
LDA on PLS latent variables (PLS-LDA).
* `X` : X-data (n, p).
* `y` : Univariate class membership (n). Must be a `Vector{String}`.
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs) to compute. Must be >= 1.
* `prior` : Type of prior probabilities for class membership. Possible values are: `:prop` (proportionnal), 
    `:unif` (uniform), or a vector (of length equal to the number of classes) giving the prior weight for each class 
    (in case of vector, it must be sorted in the same order as `mlev(y)`).
* `scal` : Boolean. If `true`, each column of `X` and Ydummy is scaled by its uncorrected standard deviation
    in the PLS computation.

LDA on PLS latent variables. The approach is as follows:

1) The training variable `y` (univariate class membership) is transformed to a dummy table (Ydummy) 
    containing nlev columns, where nlev is the number of classes present in `y`. Each column of 
    Ydummy is a dummy (0/1) variable. 
2) A multivariate PLSR (PLSR2) is run on the data {`X`, Ydummy}, returning a score matrix `T`.
3) A LDA is done on {`T`, `y`}, returning estimates of posterior probabilities (∊ [0, 1]) of class membership.
4) For a given observation, the final prediction is the class corresponding to the dummy variable for which 
    the probability estimate is the highest.

The low-level function method (i.e. having argument `weights`) requires to set as input a vector of observation 
weights. In that case, argument `prior` has no effect: the class prior probabilities (output `priors`) are always 
computed by summing the observation weights by class.

In the high-level methods (no argument `weights`), argument `prior` defines how are preliminary computed the 
observation weights (see function `pweightcla`) that are then given as input in the hidden low level method.

**Note:** For highly unbalanced classes, it may be recommended to define equal class weights ('prior = :unif'),
and to use a performance score such as `merrp`, instead of `errp`.

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
#model = plslda(; nlv, prior = :unif) 
#model = plsqda(; nlv, alpha = .1) 
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
@head transf(model, Xtest, 3)

fitm_da = fitm.fitm_da ;
typeof(fitm_da)

res = predict(model, Xtest) ;
@names res
@head res.posterior
@head res.pred
errp(res.pred, ytest)
conf(res.pred, ytest).cnt

predict(model, Xtest, 1:2).pred
summary(fitm_emb, Xtrain)
```
""" 
plslda(; kwargs...) = JchemoModel(plslda, nothing, kwargs)

function plslda(X, y; kwargs...)
    par = recovkw(ParPlsda{Q}, kwargs).par
    Q = eltype(X[1, 1])
    weights = pweightcla(Q, y; prior = par.prior)
    plslda(X, y, weights; kwargs...)
end

function plslda(X::Matrix{Q}, y::Vector{String}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: Float
    par = recovkw(ParPlsda{Q}, kwargs).par
    @assert par.nlv >= 1 "Argument 'nlv' must be in >= 1"   
    res = dummy(Q, y)
    ni = tab(y).vals
    priors = aggsumv(weights.values, y).val  # output not used, only for information
    fitm_emb = plskern(X, res.Y, weights; kwargs...)
    par.nlv = fitm_emb.par.nlv
    fitm_da = list(Lda, par.nlv)
    @inbounds for i in eachindex(fitm_da)
        fitm_da[i] = lda(vcol(fitm_emb.T, 1:i), y, weights; kwargs...)
    end
    Plsprobda(fitm_emb, fitm_da, ni, priors, res.lev, par) 
end

"""
    predict(object::Plsprobda, X)
    predict(object::Plsprobda, X, nlv::Union{Int, AbstractVector{Int}})
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
* `nlv` : Nb. LVs, or collection of nb. LVs, to consider. 
""" 
function predict(object::Plsprobda, X)
    T = transf(object.fitm_emb, X)
    res = predict(object.fitm_da[end], T)
    (pred = res.pred, posterior = res.posterior, nlv = nco(T))
end

function predict(object::Plsprobda, X, nlv::Union{Int, AbstractVector{Int}})
    X = ensure_mat(X)
    Q = eltype(X)
    Qy = eltype(object.lev)
    m = nro(X)
    a = object.par.nlv
    if isa(nlv, Int)
        nlv = max(1, min(nlv, a))
    else
        nlv = max(1, min(minimum(nlv), a)):min(maximum(nlv), a)
    end
    le_nlv = length(nlv)
    T = transf(object.fitm_emb, X)
    pred = list(Matrix{Qy}, le_nlv)
    posterior = list(Matrix{Q}, le_nlv)
    @inbounds for i in eachindex(nlv)
        znlv = nlv[i]
        res = predict(object.fitm_da[znlv], vcol(T, 1:znlv))
        pred[i] = res.pred 
        posterior[i] = res.posterior
    end 
    (pred = pred, posterior, nlv)
end


