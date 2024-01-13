"""
    dkplsrda(; kwargs...)
    dkplsrda(X, y; kwargs...)
    dkplsrda(X, Y, weights::Weight; kwargs...)
Discrimination based on direct kernel partial least 
    squares regression (KPLSR-DA).
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`). 
Keyword arguments: 
* `nlv` : Nb. latent variables (LVs) to compute.
* `kern` : Type of kernel used to compute the Gram matrices.
    Possible values are: `:krbf`, `:kpol`. See respective 
    functions `krbf` and `kpol` for their keyword arguments.
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

Same as function `plsrda` (PLSR-DA) except that 
a direct kernel PLSR (function `dkplsr`), instead of a 
PLSR (function `plskern`), is run on the Y-dummy table. 

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
kern = :krbf ; gamma = .001 
scal = true
mod = dkplsrda(; nlv,
    kern, gamma, scal) 
fit!(mod, Xtrain, ytrain)
pnames(mod)
pnames(mod.fm)
fm = mod.fm ;
fm.lev
fm.ni

@head fm.fm.fm.T
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
```
""" 
function dkplsrda(X, y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    dkplsrda(X, y, weights; kwargs...)
end

function dkplsrda(X, Y, weights::Weight; kwargs...)
    res = dummy(y)
    ni = tab(y).vals
    fm = dkplsr(X, res.Y, weights; 
        kwargs...)
    Dkplsrda(fm, res.lev, ni)
end

""" 
    transf(object::Dkplsrda, X; nlv = nothing)
Compute latent variables (LVs = scores T) from 
    a fitted model.
* `object` : The fitted model.
* `X` : Matrix (m, p) for which LVs are computed.
* `nlv` : Nb. LVs to consider.
""" 
function transf(object::Dkplsrda, X; nlv = nothing)
    transf(object.fm, X; nlv)
end

"""
    predict(object::Dkplsrda, X; nlv = nothing)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
* `nlv` : Nb. LVs, or collection of nb. LVs, to consider. 
If nothing, it is the maximum nb. LVs.
""" 
function predict(object::Dkplsrda, X; 
        nlv = nothing)
    X = ensure_mat(X)
    Q = eltype(X)
    Qy = eltype(object.lev)
    m = nro(X)
    a = size(object.fm.fm.T, 2)
    isnothing(nlv) ? nlv = a : 
        nlv = (max(minimum(nlv), 0):min(maximum(nlv), a))
    le_nlv = length(nlv)
    pred = list(Matrix{Qy}, le_nlv)
    posterior = list(Matrix{Q}, le_nlv)
    @inbounds for i = 1:le_nlv
        zp = predict(object.fm, X; 
            nlv = nlv[i]).pred
        z =  mapslices(argmax, zp; 
            dims = 2)  # if equal, argmax takes the first
        pred[i] = reshape(replacebylev2(z, 
            object.lev), m, 1)     
        posterior[i] = zp
    end 
    if le_nlv == 1
        pred = pred[1]
        posterior = posterior[1]
    end
    (pred = pred, posterior = posterior)
end