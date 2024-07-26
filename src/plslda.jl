"""
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

LDA on PLS latent variables: 
1) The training variable `y` (univariate class membership) is 
    transformed to a dummy table (Ydummy) containing nlev columns, where 
    nlev is the number of classes present in `y`. Each column of Ydummy 
    is a dummy (0/1) variable. 
2) A weighted multivariate PLS ("PLS2") is run on {`X`, Ydummy}, returning 
    a score matrix `T`.
3) A LDA is done on {`T`, `y`}. 

In these `plslda` functions:
- observation weights (argument `weights`) are used to compute the PLS scores 
    and the LDA intra-class (= "within") covariance matrix, 
- argument `prior` is used to define the usual LDA prior class probabilities. 

In the high-level version, the observation weights are automatically 
defined by the given priors: the sub-total weights by class are set 
equal to the prior probabilities. For other choices, use the low-level 
version.

## Examples
```julia
using Jchemo, JchemoData, JLD2
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
mod = model(plslda; nlv) 
#mod = model(plslda; nlv, prior = :prop) 
#mod = model(plsqda; nlv, alpha = .1) 
fit!(mod, Xtrain, ytrain)
pnames(mod)
pnames(mod.fm)
fm = mod.fm ;
fm.lev
fm.ni

fmpls = fm.fm.fmpls ;
@head fmpls.T
@head transf(mod, Xtrain)
@head transf(mod, Xtest)
@head transf(mod, Xtest; nlv = 3)

coef(fmpls)

res = predict(mod, Xtest) ;
pnames(res)
@head res.posterior
@head res.pred
errp(res.pred, ytest)
conf(res.pred, ytest).cnt

predict(mod, Xtest; nlv = 1:2).pred
summary(fmpls, Xtrain)
```
""" 
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
    fmpls = plskern(X, res.Y, weights; kwargs...)
    fmda = list(Lda, par.nlv)
    @inbounds for i = 1:par.nlv
        fmda[i] = lda(fmpls.T[:, 1:i], y, weights; kwargs...)
    end
    fm = (fmpls = fmpls, fmda = fmda)
    Plsprobda(fm, res.lev, ni, par)
end

""" 
    transf(object::Plsprobda, X; nlv = nothing)
Compute latent variables (LVs = scores T) from 
    a fitted model.
* `object` : The fitted model.
* `X` : Matrix (m, p) for which LVs are computed.
* `nlv` : Nb. LVs to consider.
""" 
function transf(object::Plsprobda, X; nlv = nothing)
    transf(object.fm.fmpls, X; nlv)
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
    a = size(object.fm.fmpls.T, 2)
    isnothing(nlv) ? nlv = a : nlv = (max(minimum(nlv), 0):min(maximum(nlv), a))
    le_nlv = length(nlv)
    pred = list(Matrix{Qy}, le_nlv)
    posterior = list(Matrix{Q}, le_nlv)
    @inbounds for i = 1:le_nlv
        znlv = nlv[i]
        T = transf(object.fm.fmpls, X; nlv = znlv)
        zres = predict(object.fm.fmda[znlv], T)
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


