"""
    mbplskdeda(Xbl, y; kwargs...)
    mbplskdeda(Xbl, y, weights::Weight; kwargs...)
Multiblock PLS-KDEDA.
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
    (the vector must be sorted in the same order as `mlev(x)`).
* Keyword arguments of function `dmkern` (bandwidth 
    definition) can also be specified here.
* `scal` : Boolean. If `true`, each column of blocks in `Xbl` 
    and `Y` is scaled by its uncorrected standard deviation 
    (before the block scaling).

This is the same principle as function `plskdeda`, for multiblock X-data.

See function `mbplslda` for examples.

""" 
function mbplskdeda(Xbl, y; kwargs...)
    par = recovkwargs(Par, kwargs)
    Q = eltype(Xbl[1][1, 1])
    weights = mweightcla(Q, y; prior = par.prior)
    mbplskdeda(Xbl, y, weights; kwargs...)
end

function mbplskdeda(Xbl, y, weights::Weight; kwargs...)
    par = recovkwargs(Par, kwargs)
    @assert par.nlv >= 1 "Argument 'nlv' must be in >= 1"   
    res = dummy(y)
    ni = tab(y).vals
    fmpls = mbplsr(Xbl, res.Y, weights; kwargs...)
    fmda = list(Kdeda, par.nlv)
    @inbounds for i = 1:par.nlv
        fmda[i] = kdeda(fmpls.T[:, 1:i], y; kwargs...)
    end
    fm = (fmpls = fmpls, fmda = fmda)
    Mbplslda(fm, res.lev, ni)
end

