"""
    dkplsqda(X, y; kwargs...)
    dkplsqda(X, y, weights::Weight; kwargs...)
DKPLS-QDA.
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs) to compute.
    Must be >= 1
* `kern` : Type of kernel used to compute the Gram matrices.
    Possible values are: `:krbf`, `:kpol`. See respective 
    functions `krbf` and `kpol` for their keyword arguments.
* `prior` : Type of prior probabilities for class 
    membership. Possible values are: `:unif` (uniform), 
    `:prop` (proportional), or a vector (of length equal to 
    the number of classes) giving the prior weight for each class 
    (in case of vector, it must be sorted in the same order as `mlev(y)`).
* `alpha` : Scalar (∈ [0, 1]) defining the continuum
    between QDA (`alpha = 0`) and LDA (`alpha = 1`).
* `scal` : Boolean. If `true`, each column of `X` 
    and Ydummy is scaled by its uncorrected standard deviation
    in the PLS computation.

Same as function `plsqda` (PLS-QDA) except that 
a direct kernel PLSR (function `dkplsr`), instead of a 
PLSR (function `plskern`), is run on the Y-dummy table. 

See function `dkplslda` for examples.
""" 
function dkplsqda(X, y; kwargs...)
    par = recovkw(ParKplsqda, kwargs).par
    Q = eltype(X[1, 1])
    weights = mweightcla(Q, y; prior = par.prior)
    dkplsqda(X, y, weights; kwargs...)
end

function dkplsqda(X, y, weights::Weight; kwargs...)
    par = recovkw(ParKplsqda, kwargs).par
    @assert par.nlv >= 1 "Argument 'nlv' must be in >= 1"   
    res = dummy(y)
    ni = tab(y).vals
    fmpls = dkplsr(X, res.Y, weights; kwargs...)
    fmda = list(Qda, par.nlv)
    @inbounds for i = 1:par.nlv
        fmda[i] = qda(fmpls.T[:, 1:i], y, weights; kwargs...)
    end
    fm = (fmpls = fmpls, fmda = fmda)
    Plsprobda(fm, res.lev, ni, par)
end



