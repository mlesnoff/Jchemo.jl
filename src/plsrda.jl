"""
    plsrda(; kwargs...)
    plsrda(X, y; kwargs...)
    plsrda(X, y, weights::Weight; kwargs...)
Discrimination based on partial least squares regression (PLSR-DA).
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g. function `mweight`). 
Keyword arguments: 
* `nlv` : Nb. latent variables (LVs) to compute.
* `prior` : Type of prior probabilities for class membership. Possible values are: `:prop` (proportionnal), 
    `:unif` (uniform), or a vector (of length equal to the number of classes) giving the prior weight for each class 
    (in case of vector, it must be sorted in the same order as `mlev(y)`).
* `scal` : Boolean. If `true`, each column of `X` and Ydummy is scaled by its uncorrected standard deviation.

This is the usual "PLSDA". The approach is as follows:

1) The training variable `y` (univariate class membership) is transformed to a dummy table (Ydummy) 
    containing nlev columns, where nlev is the number of classes present in `y`. Each column of 
    Ydummy is a dummy (0/1) variable. 
2) Then, a multivariate PLSR (PLSR2) is run on {`X`, Ydummy}, returning predictions of the dummy variables 
    (= object `posterior` returned by fuction `predict`).  These predictions can be considered as unbounded 
    estimates (i.e. eventually outside of [0, 1]) of the class membership probabilities.
3) For a given observation, the final prediction is the class corresponding to the dummy variable for which 
    the probability estimate is the highest.

The low-level method (i.e. having argument `weights`) of the function allows to set any vector of observation weights 
to be used in the intermediate computations. In the high-level methods (no argument `weights`), they are automatically 
computed from the argument `prior` value: for each class, the total of the observation weights is set equal to 
the prior probability corresponding to the class.

**Note:** For highly unbalanced classes, it may be recommended to set 'prior = :unif' when using the function
(and to use a score such as `merrp` instead of `errp` when evaluating the perfomance).

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
model = plsrda(; nlv) 
#model = plsrda(; nlv, prior = :unif) 
fit!(model, Xtrain, ytrain)
@names model
fitm = model.fitm ;
typeof(fitm)
@names fitm
typeof(fitm.fitm) 
@names fitm.fitm
fitm.lev
fitm.ni
aggsumv(fitm.fitm.weights.w, ytrain)

@head fitm.fitm.T
@head transf(model, Xtrain)
@head transf(model, Xtest)
@head transf(model, Xtest; nlv = 3)

coef(fitm.fitm)

res = predict(model, Xtest) ;
@names res
@head res.posterior
@head res.pred
errp(res.pred, ytest)
conf(res.pred, ytest).cnt

predict(model, Xtest; nlv = 1:2).pred
summary(fitm.fitm, Xtrain)
```
"""
plsrda(; kwargs...) = JchemoModel(plsrda, nothing, kwargs)

function plsrda(X, y; kwargs...)
    par = recovkw(ParPlsda, kwargs).par
    Q = eltype(X[1, 1])
    weights = mweightcla(Q, y; prior = par.prior)
    plsrda(X, y, weights; kwargs...)
end

function plsrda(X, y, weights::Weight; kwargs...)
    par = recovkw(ParPlsda, kwargs).par
    Q = eltype(X[1, 1])
    res = dummy(y)
    ni = tab(y).vals
    fitm = plskern(X, res.Y, weights; kwargs...)
    Plsrda(fitm, res.lev, ni, par)
end

""" 
    transf(object::Plsrda, X; nlv = nothing)
Compute latent variables (LVs; = scores) from 
    a fitted model.
* `object` : The fitted model.
* `X` : X-data (m, p) for which LVs are computed.
* `nlv` : Nb. LVs to consider.
""" 
function transf(object::Plsrda, X; nlv = nothing)
    transf(object.fitm, X; nlv)
end

"""
    predict(object::Plsrda, X; nlv = nothing)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
* `nlv` : Nb. LVs, or collection of nb. LVs, to consider. 
""" 
function predict(object::Plsrda, X; nlv = nothing)
    X = ensure_mat(X)
    Q = eltype(X)
    Qy = eltype(object.lev)
    m = nro(X)
    a = nco(object.fitm.T)
    isnothing(nlv) ? nlv = a : nlv = min(a, minimum(nlv)):min(a, maximum(nlv))
    le_nlv = length(nlv)
    pred = list(Matrix{Qy}, le_nlv)
    posterior = list(Matrix{Q}, le_nlv)
    @inbounds for i in eachindex(nlv)
        zpred = predict(object.fitm, X; nlv = nlv[i]).pred
        #if softmax
        #    @inbounds for j = 1:m
        #        zpred[j, :] .= mweight(exp.(zpred[j, :]))
        #   end
        #end
        z =  mapslices(argmax, zpred; dims = 2)  # if equal, argmax takes the first
        pred[i] = reshape(recod_indbylev(z, object.lev), m, 1)     
        posterior[i] = zpred
    end 
    if le_nlv == 1
        pred = pred[1]
        posterior = posterior[1]
    end
    (pred = pred, posterior = posterior)
end

