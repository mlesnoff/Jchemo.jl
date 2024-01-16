"""
    mbplsrda(; kwargs...)
    mbplsrda(Xbl, y; kwargs...)
    mbplsrda(Xbl, y, weights::Weight; kwargs...)
    mbplsr!(Xbl::Matrix, y, weights::Weight; kwargs...)
Discrimination based on multiblock partial least squares 
    regression (MBPLSR-DA).
* `Xbl` : List of blocks (vector of matrices) of X-data 
    Typically, output of function `mblock` from (n, p) data.  
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
    * `nlv` : Nb. latent variables (LVs = scores T) to compute.
    * `bscal` : Type of block scaling. Possible values are:
        `:none`, `:frob`. See functions `blockscal`.
    * `scal` : Boolean. If `true`, each column of blocks in `Xbl` 
        is scaled by its uncorrected standard deviation 
        (before the block scaling).

This is the same principle as function `plsrda`, for multiblock X-data.

## Examples
```julia
using JLD2, CairoMakie, JchemoData
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/forages2.jld2")
@load db dat
pnames(dat)
X = dat.X
Y = dat.Y
tab(Y.typ)
s = Bool.(Y.test)
Xtrain = rmrow(X, s)
ytrain = rmrow(Y.typ, s)
Xtest = X[s, :]
ytest = Y.typ[s]
ntrain = nro(Xtrain)
ntest = nro(Xtest)
ntot = ntrain + ntest
(ntot = ntot, ntrain, ntest)
wlst = names(X)
wl = parse.(Float64, wlst)
#plotsp(X, wl; nsamp = 20).f
##
listbl = [1:350, 351:700]
Xbl_train = mblock(Xtrain, listbl)
Xbl_test = mblock(Xtest, listbl) 

nlv = 15
scal = false
#scal = true
bscal = :none
#bscal = :frob
mod = mbplsrda(; nlv, bscal, scal)
fit!(mod, Xbl_train, ytrain) 
pnames(mod) 

@head mod.fm.fm.T 
@head transf(mod, Xbl_train)

@head transf(mod, Xbl_test)

res = predict(mod, Xbl_test) ; 
@head res.pred 
@show errp(res.pred, ytest)
confusion(res.pred, ytest).cnt

predict(mod, Xbl_test; nlv = 1:2).pred
```
"""
function mbplsrda(Xbl, y; kwargs...)
    Q = eltype(Xbl[1][1, 1])
    n = nro(Xbl[1])
    weights = mweight(ones(Q, n))
    mbplsrda(Xbl, y, weights; kwargs...)
end

function mbplsrda(Xbl, y, weights::Weight; kwargs...)
    res = dummy(y)
    ni = tab(y).vals
    fm = mbplsr(Xbl, res.Y, weights; kwargs...)
    Mbplsrda(fm, res.lev, ni) 
end

""" 
    transf(object::Mbplsrda, Xbl; nlv = nothing)
Compute latent variables (LVs = scores T) from a fitted model.
* `object` : The fitted model.
* `Xbl` : A list of blocks (vector of matrices) 
    of X-data for which LVs are computed.
* `nlv` : Nb. LVs to compute.
""" 
function transf(object::Mbplsrda, Xbl; nlv = nothing)
    transf(object.fm, Xbl; nlv)
end

"""
    predict(object::Mbplsrda, Xbl; nlv = nothing)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `Xbl` : A list of blocks (vector of matrices) 
    of X-data for which predictions are computed.
* `nlv` : Nb. LVs, or collection of nb. LVs, to consider. 
""" 
function predict(object::Mbplsrda, Xbl; nlv = nothing)
    Q = eltype(Xbl[1][1, 1])
    Qy = eltype(object.lev)
    m = nro(Xbl[1])
    a = nco(object.fm.T)
    isnothing(nlv) ? nlv = a : nlv = (max(minimum(nlv), 0):min(maximum(nlv), a))
    le_nlv = length(nlv)
    pred = list(Matrix{Qy}, le_nlv)
    posterior = list(Matrix{Q}, le_nlv)
    @inbounds for i = 1:le_nlv
        zpred = predict(object.fm, Xbl; nlv = nlv[i]).pred
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