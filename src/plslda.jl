"""
    plslda(; kwargs...)
    plslda(X, y; kwargs...)
    plslda(X, Y, weights::Weight; kwargs...)
PLS-LDA.
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs) to compute.
    Must be >= 1.
* `prior` : Type of prior probabilities for class 
    membership. Possible values are: `:unif` (uniform), 
    `:prop` (proportional).
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

LDA on PLS latent variables. The training variable `y` 
(univariate class membership) is transformed to a dummy table 
(Ydummy) containing nlev columns, where nlev is the number of 
classes present in `y`. Each column of Ydummy is a dummy (0/1) 
variable. Then, a PLSR2 (i.e. multivariate) is run on 
{`X`, Ydummy}, returning a score matrix `T`. Finally, a LDA 
is done on {`T`, `y`}. 

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
mod = plslda(; nlv) 
#mod = plslda(; nlv, prior = :prop) 
#mod = plsqda(; nlv) 
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
confusion(res.pred, ytest).cnt

predict(mod, Xtest; nlv = 1:2).pred
summary(fmpls, Xtrain)
```
""" 
function plslda(X, y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    plslda(X, y, weights; 
        kwargs...)
end

function plslda(X, Y, weights::Weight; kwargs...)
    par = recovkwargs(Par, kwargs)
    @assert par.nlv >= 1 "Argument 'nlv' must be in >= 1"   
    res = dummy(y)
    ni = tab(y).vals
    fmpls = plskern(X, res.Y, weights; 
        kwargs...)
    fmda = list(Lda, par.nlv)
    @inbounds for i = 1:par.nlv
        fmda[i] = lda(fmpls.T[:, 1:i], y, 
            weights; kwargs...)
    end
    fm = (fmpls = fmpls, fmda = fmda)
    Plslda(fm, res.lev, ni)
end

""" 
    transf(object::Plslda, X; 
        nlv = nothing)
Compute latent variables (LVs = scores T) from 
    a fitted model.
* `object` : The fitted model.
* `X` : Matrix (m, p) for which LVs are computed.
* `nlv` : Nb. LVs to consider.
""" 
function transf(object::Plslda, X; 
        nlv = nothing)
    transf(object.fm.fmpls, X; nlv)
end

"""
    predict(object::Plslda, X; nlv = nothing)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
* `nlv` : Nb. LVs, or collection of nb. LVs, to consider. 
""" 
function predict(object::Plslda, X; 
        nlv = nothing)
    X = ensure_mat(X)
    Q = eltype(X)
    Qy = eltype(object.lev)
    m = nro(X)
    a = size(object.fm.fmpls.T, 2)
    isnothing(nlv) ? nlv = a : 
        nlv = (max(minimum(nlv), 0):min(maximum(nlv), a))
    le_nlv = length(nlv)
    pred = list(Matrix{Qy}, le_nlv)
    posterior = list(Matrix{Q}, le_nlv)
    @inbounds for i = 1:le_nlv
        znlv = nlv[i]
        T = transf(object.fm.fmpls, X; 
            nlv = znlv)
        zres = predict(object.fm.fmda[znlv], T)
        z =  mapslices(argmax, zres.posterior; 
            dims = 2) 
        pred[i] = reshape(replacebylev2(z, 
            object.lev), m, 1)
        posterior[i] = zres.posterior
    end 
    if le_nlv == 1
        pred = pred[1]
        posterior = posterior[1]
    end
    (pred = pred, posterior = posterior)
end





