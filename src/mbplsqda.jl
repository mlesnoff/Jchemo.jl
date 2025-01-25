"""
    mbplsqda(; kwargs...)
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
    (in case of vector, it must be sorted in the same order as `mlev(y)`).
* `alpha` : Scalar (∈ [0, 1]) defining the continuum
    between QDA (`alpha = 0`) and LDA (`alpha = 1`).
* `scal` : Boolean. If `true`, each column of blocks in `Xbl` 
    and Ydummy is scaled by its uncorrected standard deviation 
    (before the block scaling) in the MBPLS computation.

The method is as follows:

1) The training variable `y` (univariate class membership) is 
    transformed to a dummy table (Ydummy) containing nlev columns, 
    where nlev is the number of classes present in `y`. Each column of 
    Ydummy is a dummy (0/1) variable. 
2) A multivariate MBPLSR (MBPLSR2) is run on {`X`, Ydummy}, returning 
    a score matrix `T`.
3) A QDA (possibly with continuum) is done on {`T`, `y`}, returning estimates
    of posterior probabilities (∊ [0, 1]) of class membership.
4) For a given observation, the final prediction is the class 
    corresponding to the dummy variable for which the probability 
    estimate is the highest.

In the high-level version of the present functions, the observation 
weights are automatically defined by the given priors (argument `prior`): 
the sub-totals by class of the observation weights are set equal to the prior 
probabilities. The low-level version (argument `weights`) allows to implement 
other choices.

See function `mbplslda` for examples.

""" 
mbplsqda(; kwargs...) = JchemoModel(mbplsqda, nothing, kwargs)

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
    embfitm = mbplsr(Xbl, res.Y, weights; kwargs...)
    dafitm = list(Qda, par.nlv)
    @inbounds for i = 1:par.nlv
        dafitm[i] = qda(vcol(embfitm.T, 1:i), y, weights; kwargs...)
    end
    fitm = (embfitm = embfitm, dafitm = dafitm)
    Mbplsprobda(fitm, res.lev, ni, par)
end

