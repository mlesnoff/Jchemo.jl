"""
    mbplslda(; kwargs...)
    mbplslda(Xbl, y; kwargs...)
    mbplslda(Xbl, y, weights::Weight; kwargs...)
Multiblock PLS-LDA.
* `Xbl` : List of blocks (vector of matrices) of X-data. Typically, output of function `mblock` from data (n, p).  
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs; = scores) to compute.
* `bscal` : Type of block scaling. See function `blockscal` for possible values.
* `prior` : Type of prior probabilities for class membership. Possible values are: `:prop` (proportionnal), 
    `:unif` (uniform), or a vector (of length equal to the number of classes) giving the prior weight for each class 
    (in case of vector, it must be sorted in the same order as `mlev(y)`).
* `scal` : Boolean. If `true`, each column of blocks in `Xbl` and Ydummy is scaled by its uncorrected standard deviation 
    (before the block scaling) in the MBPLS computation.

The approach is as follows:

1) The training variable `y` (univariate class membership) is transformed to a dummy table (Ydummy) 
    containing nlev columns, where nlev is the number of classes present in `y`. Each column of 
    Ydummy is a dummy (0/1) variable. 
2) A multivariate MBPLSR (MBPLSR2) is run on {`X`, Ydummy}, returning 
    a score matrix `T`.
3) A LDA is done on {`T`, `y`}, returning estimates of posterior probabilities
    (∊ [0, 1]) of class membership.
4) For a given observation, the final prediction is the class corresponding to the dummy variable for which 
    the probability estimate is the highest.

The low-level method (i.e. having argument `weights`) of the function allows to set any vector of observation weights 
to be used in the intermediate computations. In the high-level methods (no argument `weights`), they are automatically 
computed from the argument `prior` value: for each class, the total of the observation weights is set equal to 
the prior probability corresponding to the class.

**Note:** For highly unbalanced classes, it may be recommended to set 'prior = :unif' when using the function
(and to use a score such as `merrp` instead of `errp` when evaluating the perfomance).

## Examples
```julia
using Jchemo, JLD2, CairoMakie, JchemoData
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/forages2.jld2")
@load db dat
@names dat
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
model = mbplslda(; nlv, bscal, scal)
#model = mbplsqda(; nlv, bscal, alpha = .5, scal)
#model = mbplskdeda(; nlv, bscal, scal)
fit!(model, Xbltrain, ytrain) 
@names model 

@head transf(model, Xbltrain)
@head transf(model, Xbltest)

res = predict(model, Xbltest) ; 
@head res.pred 
@show errp(res.pred, ytest)
conf(res.pred, ytest).cnt

predict(model, Xbltest; nlv = 1:2).pred
```
""" 
mbplslda(; kwargs...) = JchemoModel(mbplslda, nothing, kwargs)

function mbplslda(Xbl, y; kwargs...)
    par = recovkw(ParMbplsda, kwargs).par
    Q = eltype(Xbl[1][1, 1])
    weights = mweightcla(Q, y; prior = par.prior)
    mbplslda(Xbl, y, weights; kwargs...)
end

function mbplslda(Xbl, y, weights::Weight; kwargs...)
    par = recovkw(ParMbplsda, kwargs).par
    @assert par.nlv >= 1 "Argument 'nlv' must be in >= 1"   
    res = dummy(y)
    ni = tab(y).vals
    fitm_emb = mbplsr(Xbl, res.Y, weights; kwargs...)
    fitm_da = list(Lda, par.nlv)
    @inbounds for i = 1:par.nlv
        fitm_da[i] = lda(vcol(fitm_emb.T, 1:i), y, weights; kwargs...)
    end
    fitm = (fitm_emb = fitm_emb, fitm_da = fitm_da)
    Mbplsprobda(fitm, res.lev, ni, par)
end

""" 
    transf(object::Mbplsprobda, Xbl; nlv = nothing)
Compute latent variables (LVs; = scores) from 
    a fitted model.
* `object` : The fitted model.
* `Xbl` : A list of blocks (vector of matrices) 
    of X-data for which LVs are computed.
* `nlv` : Nb. LVs to consider.
""" 
function transf(object::Mbplsprobda, Xbl; nlv = nothing)
    transf(object.fitm_emb, Xbl; nlv)
end

"""
    predict(object::Mbplsprobda, Xbl; nlv = nothing)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `Xbl` : A list of blocks (vector of matrices) 
    of X-data for which predictions are computed.
* `nlv` : Nb. LVs, or collection of nb. LVs, to consider. 
""" 
function predict(object::Mbplsprobda, Xbl; nlv = nothing)
    Q = eltype(Xbl[1][1, 1])
    Qy = eltype(object.lev)
    m = nro(Xbl[1])
    a = size(object.fitm_emb.T, 2)
    isnothing(nlv) ? nlv = a : nlv = min(a, minimum(nlv)):min(a, maximum(nlv))
    le_nlv = length(nlv)
    pred = list(Matrix{Qy}, le_nlv)
    posterior = list(Matrix{Q}, le_nlv)
    @inbounds for i in eachindex(nlv)
        znlv = nlv[i]
        T = transf(object.fitm_emb, Xbl; nlv = znlv)
        zres = predict(object.fitm_da[znlv], T)
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





