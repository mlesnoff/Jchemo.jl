"""
    mbplsrda(Xbl, y; kwargs...)
    mbplsrda(Xbl, y, weights::Weight; kwargs...)
Discrimination based on multiblock partial least squares 
    regression (MBPLSR-DA).
* `Xbl` : List of blocks (vector of matrices) of X-data 
    Typically, output of function `mblock` from (n, p) data.  
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs = scores T) to compute.
* `bscal` : Type of block scaling. See function `blockscal`
    for possible values.
* `prior` : Type of prior probabilities for class 
    membership. Possible values are: `:unif` (uniform), 
    `:prop` (proportional), or a vector (of length equal to 
    the number of classes) giving the prior weight for each class 
    (in case of vector, it must be sorted in the same order as `mlev(y)`).
* `scal` : Boolean. If `true`, each column of blocks in `Xbl` 
    and Ydummy is scaled by its uncorrected standard deviation 
    (before the block scaling).

The method is as follows:

1) The training variable `y` (univariate class membership) is 
    transformed to a dummy table (Ydummy) containing nlev columns, 
    where nlev is the number of classes present in `y`. Each column of 
    Ydummy is a dummy (0/1) variable. 
2) Then, a multivariate MBPLSR (MBPLSR2) is run on {`X`, Ydummy}, returning 
    predictions of the dummy variables (= object `posterior` returned by 
    fuction `predict`).  These predictions can be considered as unbounded estimates 
    (i.e. eventuall outside of [0, 1]) of the class membership probabilities.
3) For a given observation, the final prediction is the class 
    corresponding to the dummy variable for which the probability 
    estimate is the highest.

In the high-level version of the present functions, the observation 
weights are automatically defined by the given priors (argument `prior`): 
the sub-totals by class of the observation weights are set equal to the prior 
probabilities. The low-level version (argument `weights`) allows to implement 
other choices.

## Examples
```julia
using Jchemo, JLD2, CairoMakie, JchemoData
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
Xbltrain = mblock(Xtrain, listbl)
Xbltest = mblock(Xtest, listbl) 

nlv = 15
scal = false
#scal = true
bscal = :none
#bscal = :frob
mod = model(mbplsrda; nlv, bscal, scal)
fit!(mod, Xbltrain, ytrain) 
pnames(mod) 

@head mod.fm.fm.T 
@head transf(mod, Xbltrain)
@head transf(mod, Xbltest)

res = predict(mod, Xbltest) ; 
@head res.pred 
@show errp(res.pred, ytest)
conf(res.pred, ytest).cnt

predict(mod, Xbltest; nlv = 1:2).pred
```
"""
function mbplsrda(Xbl, y; kwargs...)
    par = recovkw(ParMbplsda, kwargs).par
    Q = eltype(Xbl[1][1, 1])
    weights = mweightcla(Q, y; prior = par.prior)
    mbplsrda(Xbl, y, weights; kwargs...)
end

function mbplsrda(Xbl, y, weights::Weight; kwargs...)
    par = recovkw(ParMbplsda, kwargs).par
    res = dummy(y)
    ni = tab(y).vals
    fm = mbplsr(Xbl, res.Y, weights; kwargs...)
    Mbplsrda(fm, res.lev, ni, par) 
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
        pred[i] = reshape(recod_indbylev(z, object.lev), m, 1)     
        posterior[i] = zpred
    end 
    if le_nlv == 1
        pred = pred[1]
        posterior = posterior[1]
    end
    (pred = pred, posterior = posterior)
end