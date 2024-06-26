"""
    kplskdeda(X, y; kwargs...)
    kplskdeda(X, y, weights::Weight; kwargs...)
KPLS-KDEDA.
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
    (the vector must be sorted in the same order as `mlev(y)`).
* Keyword arguments of function `dmkern` (bandwidth 
    definition) can also be specified here.
* `alpha` : Scalar (∈ [0, 1]) defining the continuum
    between QDA (`alpha = 0`) and LDA (`alpha = 1`).
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

Same as function `plskdeda` (PLS-KDEDA) except that 
a kernel PLSR (function `kplsr`), instead of a 
PLSR (function `plskern`), is run on the Y-dummy table. 

See function `kplslda` for examples.
""" 
function kplskdeda(X, y; kwargs...)
    par = recovkwargs(Par, kwargs)
    Q = eltype(X[1, 1])
    weights = mweightcla(Q, y; prior = par.prior)
    kplskdeda(X, y, weights; kwargs...)
end

function kplskdeda(X, y, weights::Weight; kwargs...)
    par = recovkwargs(Par, kwargs)
    @assert par.nlv >= 1 "Argument 'nlv' must be in >= 1"   
    res = dummy(y)
    ni = tab(y).vals
    fmpls = kplsr(X, res.Y, weights; kwargs...)
    fmda = list(Kdeda, par.nlv)
    @inbounds for i = 1:par.nlv
        fmda[i] = kdeda(fmpls.T[:, 1:i], y; kwargs...)
    end
    fm = (fmpls = fmpls, fmda = fmda)
    Plslda(fm, res.lev, ni)
end



