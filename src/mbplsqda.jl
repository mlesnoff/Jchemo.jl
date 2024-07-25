"""
    mbplsqda(Xbl, y; kwargs...)
    mbplsqda(Xbl, y, weights::Weight; kwargs...)
Multiblock PLS-QDA.
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
    (the vector must be sorted in the same order as `mlev(y)`).
* `alpha` : Scalar (∈ [0, 1]) defining the continuum
    between QDA (`alpha = 0`) and LDA (`alpha = 1`).
* `scal` : Boolean. If `true`, each column of blocks in `Xbl` 
    and Ydummy is scaled by its uncorrected standard deviation 
    (before the block scaling) in the MBPLS computation.

This is the same principle as function `plsqda`, for multiblock X-data.

See function `mbplslda` for examples.

""" 
function mbplsqda(Xbl, y; kwargs...)
    par = recovkw(ParMbplsqda, kwargs).par
    Q = eltype(Xbl[1][1, 1])
    weights = mweightcla(Q, y; prior = par.prior)
    mbplsqda(Xbl, y, weights; kwargs...)
end

function mbplsqda(Xbl, y, weights::Weight; kwargs...)
    par = recovkw(ParMbplsqda, kwargs).par
    @assert par.nlv >= 1 "Argument 'nlv' must be in >= 1"   
    res = dummy(y)
    ni = tab(y).vals
    fmpls = mbplsr(Xbl, res.Y, weights; kwargs...)
    fmda = list(Qda, par.nlv)
    @inbounds for i = 1:par.nlv
        fmda[i] = qda(fmpls.T[:, 1:i], y, weights; kwargs...)
    end
    fm = (fmpls = fmpls, fmda = fmda)
    Mbplsprobda(fm, res.lev, ni, par)
end

