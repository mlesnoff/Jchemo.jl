"""
    plsrda(; kwargs...)
    plsrda(X, y; kwargs...)
    plsrda(X, y, weights::ProbabilityWeights; kwargs...)
Discrimination based on partial least squares regression (PLSR-DA).
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`). 
Keyword arguments: 
* `nlv` : Nb. latent variables (LVs) to compute.
* `prior` : Type of prior probabilities for class membership. Possible values are: `:prop` (proportionnal), 
    `:unif` (uniform), or a vector (of length equal to the number of classes) giving the prior weight for each class 
    (in case of vector, it must be sorted in the same order as `mlev(y)`).
* `scal` : Boolean. If `true`, each column of `X` and Ydummy is scaled by its uncorrected standard deviation.

This is the usual and simplest "PLSDA". The approach is as follows:

1) The training variable `y` (univariate class membership) is transformed to a dummy table (Ydummy) 
    containing nlev columns, where nlev is the number of classes present in `y`. Each column of 
    Ydummy is a dummy (0/1) variable. 
2) Then, a multivariate PLSR (PLSR2) is run on the data {`X`, Ydummy}, returning predictions of the dummy variables 
    (= object `posterior` returned by fuction `predict`).  These predictions can be considered as unbounded 
    estimates (i.e. eventually outside of [0, 1]) of the class membership probabilities.
3) For a given observation, the final prediction is the class corresponding to the dummy variable for which 
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
using Jchemo, JchemoData, JLD2, CairoMakie
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
model = plsrda(; nlv) 
#model = plsrda(; nlv, prior = :unif) 
fit!(model, Xtrain, ytrain)
@names model
fitm = model.fitm ;
typeof(fitm)
@names fitm
typeof(fitm.fitm_emb) 
@names fitm.fitm_emb

fitm.lev
fitm.ni
fitm.priors
aggsumv(fitm.fitm_emb.weights.values, ytrain)

@head transf(model, Xtrain)
@head fitm.fitm_emb.T

@head transf(model, Xtest)
@head transf(model, Xtest, 3)

coef(fitm.fitm_emb)

res = predict(model, Xtest) ;
@names res
@head res.posterior
@head res.pred
errp(res.pred, ytest)
conf(res.pred, ytest).cnt

predict(model, Xtest; nlv = 1:2).pred

summary(fitm.fitm_emb, Xtrain)
```
"""
plsrda(; kwargs...) = JchemoModel(plsrda, nothing, kwargs)

function plsrda(X, y; kwargs...)
    par = recovkw(ParPlsda, kwargs).par
    Q = eltype(X[1, 1])
    weights = pweightcla(Q, y; prior = par.prior)
    plsrda(X, y, weights; kwargs...)
end

function plsrda(X, y, weights::ProbabilityWeights; kwargs...)
    par = recovkw(ParPlsda, kwargs).par
    res = dummy(y)
    ni = tab(y).vals
    priors = aggsumv(weights.values, vec(y)).val  # output not used, only for information
    fitm_emb = plskern(X, res.Y, weights; kwargs...)
    par.nlv = fitm_emb.par.nlv
    Plsrda(fitm_emb, ni, priors, res.lev, par)
end

""" 
    transf(object::Union{Plsrda, Plsprobda}, X, nlv::Int)
Compute latent variables (LVs; = scores) from a fitted model.
* `object` : The fitted model.
* `X` : X-data (m, p) for which LVs are computed.
* `nlv` : Nb. LVs to consider.
""" 
function transf(object::Union{Plsrda, Plsprobda}, X, nlv::Int)
    transf(object.fitm_emb, X; nlv)
end

"""
    predict(object::Plsrda, X; nlv::Union{Int, AbstractVector{Int}})
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
* `nlv` : Nb. LVs, or collection of nb. LVs, to consider. 
""" 
function predict(object::Plsrda, X; nlv::Union{Int, AbstractVector{Int}})
    X = ensure_mat(X)
    m = nro(X)
    Qy = eltype(object.lev)
    pred_fitm_emb = predict(object.fitm_emb, X; nlv)
    nlv = pred_fitm_emb.nlv
    le_nlv = length(nlv)
    if le_nlv == 1
        post = [pred_fitm_emb.pred]
    else
        post = pred_fitm_emb.pred
    end
    pred = list(Matrix{Qy}, le_nlv)
    @inbounds for i in eachindex(nlv)
        v =  mapslices(argmax, post[i]; dims = 2)  # if equal, argmax takes the first
        pred[i] = reshape(recod_indbylev(v, object.lev), m, 1)
    end 
    if le_nlv == 1
        pred = pred[1]
        post = post[1]
    end
    (pred = pred, posterior = post)
end

