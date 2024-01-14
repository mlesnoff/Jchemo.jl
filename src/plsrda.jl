"""
    plsrda(; kwargs...)
    plsrda(X, y; kwargs...)
    plsrda(X, y, weights::Weight; kwargs...)
Discrimination based on partial least squares 
    regression (PLSR-DA).
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`). 
Keyword arguments: 
* `nlv` : Nb. latent variables (LVs) to compute.
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

This is the usual "PLSDA". The training variable `y` 
(univariate class membership) is transformed to a dummy table 
(Ydummy) containing nlev columns, where nlev is the number of 
classes present in `y`. Each column of Ydummy is a dummy (0/1) 
variable. Then, a PLSR2 (i.e. multivariate) is run on 
{`X`, Ydummy}, returning predictions of the dummy 
variables (= object `posterior` returned by fuction `predict`).  
These predictions can be considered as unbounded estimates (i.e. 
eventuall outside of [0, 1]) of the class membership probabilities. 
For a given observation, the final prediction is the class 
corresponding to the dummy variable for which the probability 
estimate is the highest.

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
mod = plsrda(; nlv) 
fit!(mod, Xtrain, ytrain)
pnames(mod)
pnames(mod.fm)
fm = mod.fm ;
fm.lev
fm.ni

@head fm.fm.T
@head transf(mod, Xtrain)
@head transf(mod, Xtest)
@head transf(mod, Xtest; nlv = 3)

coef(fm.fm)

res = predict(mod, Xtest) ;
pnames(res)
@head res.posterior
@head res.pred
errp(res.pred, ytest)
confusion(res.pred, ytest).cnt

predict(mod, Xtest; nlv = 1:2).pred
summary(fm.fm, Xtrain)
```
"""
function plsrda(X, y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    plsrda(X, y, weights; kwargs...)
end

function plsrda(X, y, weights::Weight; kwargs...)
    res = dummy(y)
    ni = tab(y).vals
    fm = plskern(X, res.Y, weights; kwargs...)
    Plsrda(fm, res.lev, ni)
end

""" 
    transf(object::Plsrda, X; nlv = nothing)
Compute latent variables (LVs = scores T) from 
    a fitted model.
* `object` : The fitted model.
* `X` : X-data (m, p) for which LVs are computed.
* `nlv` : Nb. LVs to consider.
""" 
function transf(object::Plsrda, X; nlv = nothing)
    transf(object.fm, X; nlv)
end

"""
    predict(object::Plsrda, X; nlv = nothing)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
* `nlv` : Nb. LVs, or collection of nb. LVs, 
    to consider. 
""" 
function predict(object::Plsrda, X; nlv = nothing)
    X = ensure_mat(X)
    Q = eltype(X)
    Qy = eltype(object.lev)
    m = nro(X)
    a = size(object.fm.T, 2)
    isnothing(nlv) ? nlv = a : nlv = (max(minimum(nlv), 0):min(maximum(nlv), a))
    le_nlv = length(nlv)
    pred = list(Matrix{Qy}, le_nlv)
    posterior = list(Matrix{Q}, le_nlv)
    @inbounds for i = 1:le_nlv
        zpred = predict(object.fm, X; 
            nlv = nlv[i]).pred
        #if softmax
        #    @inbounds for j = 1:m
        #        zpred[j, :] .= mweight(exp.(zpred[j, :]))
        #   end
        #end
        z =  mapslices(argmax, zpred; dims = 2)  # if equal, argmax takes the first
        pred[i] = reshape(replacebylev2(z, object.lev), m, 1)     
        posterior[i] = zpred
    end 
    if le_nlv == 1
        pred = pred[1]
        posterior = posterior[1]
    end
    (pred = pred, posterior = posterior)
end

