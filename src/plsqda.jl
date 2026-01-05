"""
    plsqda(; kwargs...)
    plsqda(X, y; kwargs...)
    plsqda(X, y, weights::Weight; kwargs...)
QDA on PLS latent variables (PLS-QDA) with continuum.
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs) to compute. Must be >= 1.
* `prior` : Type of prior probabilities for class membership. Possible values are: `:prop` (proportionnal), 
    `:unif` (uniform), or a vector (of length equal to the number of classes) giving the prior weight for each class 
    (in case of vector, it must be sorted in the same order as `mlev(y)`).
* `alpha` : Scalar (∈ [0, 1]) defining the continuum between QDA (`alpha = 0`) and LDA (`alpha = 1`).
* `scal` : Boolean. If `true`, each column of `X` and Ydummy is scaled by its uncorrected standard deviation
    in the PLS computation.

QDA on PLS latent variables. The approach is as follows:

1) The training variable `y` (univariate class membership) is transformed to a dummy table (Ydummy) 
    containing nlev columns, where nlev is the number of classes present in `y`. Each column of 
    Ydummy is a dummy (0/1) variable. 
2) A multivariate PLSR (PLSR2) is run on the data {`X`, Ydummy}, returning a score matrix `T`.
3) A QDA (possibly with continuum) is done on {`T`, `y`}, returning estimates of posterior probabilities (∊ [0, 1]) 
    of class membership.
4) For a given observation, the final prediction is the class corresponding to the dummy variable for which 
    the probability estimate is the highest.

The low-level function method (i.e. having argument `weights`) requires to set as input a vector of observation 
weights. In that case, argument `prior` has no effect: the class prior probabilities (output `priors`) are always 
computed by summing the observation weights by class.

In the high-level methods (no argument `weights`), argument `prior` defines how are preliminary computed the 
observation weights (see function `mweightcla`) that are then given as input in the hidden low level method.

**Note:** For highly unbalanced classes, it may be recommended to define equal class weights ('prior = :unif'),
and to use a performance score such as `merrp`, instead of `errp`.

See function `plslda` for examples.
""" 
plsqda(; kwargs...) = JchemoModel(plsqda, nothing, kwargs)

function plsqda(X, y; kwargs...)
    par = recovkw(ParPlsqda, kwargs).par
    Q = eltype(X[1, 1])
    weights = mweightcla(Q, y; prior = par.prior)
    plsqda(X, y, weights; kwargs...)
end

function plsqda(X, y, weights::Weight; kwargs...)
    par = recovkw(ParPlsqda, kwargs).par
    @assert par.nlv >= 1 "Argument 'nlv' must be in >= 1"   
    res = dummy(y)
    ni = tab(y).vals
    priors = aggsumv(weights.w, vec(y)).val  # output not used, only for information
    fitm_emb = plskern(X, res.Y, weights; kwargs...)
    fitm_da = list(Qda, par.nlv)
    @inbounds for i = 1:par.nlv
        fitm_da[i] = qda(vcol(fitm_emb.T, 1:i), y, weights; kwargs...)
    end
    Plsprobda(fitm_emb, fitm_da, ni, priors, res.lev, par) 
end


