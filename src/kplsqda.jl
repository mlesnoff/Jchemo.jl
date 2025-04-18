"""
    kplsqda(; kwargs...)
    kplsqda(X, y; kwargs...)
    kplsqda(X, y, weights::Weight; kwargs...)
KPLS-QDA.
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs) to compute.
    Must be >= 1
* `kern` : Type of kernel used to compute the Gram matrices.
    Possible values are: `:krbf`, `:kpol`. See respective 
    functions `krbf` and `kpol` for their keyword arguments.
* `prior` : Type of prior probabilities for class membership. Possible values are: `:prop` (proportionnal), 
    `:unif` (uniform), or a vector (of length equal to the number of classes) giving the prior weight for each class 
    (in case of vector, it must be sorted in the same order as `mlev(y)`).
* `alpha` : Scalar (∈ [0, 1]) defining the continuum
    between QDA (`alpha = 0`) and LDA (`alpha = 1`).
* `scal` : Boolean. If `true`, each column of `X` and Ydummy is scaled by its uncorrected standard deviation
    in the PLS computation.

Same as function `plsqda` (PLS-QDA) except that 
a kernel PLSR (function `kplsr`), instead of a 
PLSR (function `plskern`), is run on the Y-dummy table. 

See function `kplslda` for examples.
""" 
kplsqda(; kwargs...) = JchemoModel(kplsqda, nothing, kwargs)

function kplsqda(X, y; kwargs...)
    par = recovkw(ParKplsqda, kwargs).par
    Q = eltype(X[1, 1])
    weights = mweightcla(Q, y; prior = par.prior)
    kplsqda(X, y, weights; kwargs...)
end

function kplsqda(X, y, weights::Weight; kwargs...)
    par = recovkw(ParKplsqda, kwargs).par
    @assert par.nlv >= 1 "Argument 'nlv' must be in >= 1"   
    res = dummy(y)
    ni = tab(y).vals
    embfitm = kplsr(X, res.Y, weights; kwargs...)
    dafitm = list(Qda, par.nlv)
    @inbounds for a = 1:par.nlv
        dafitm[a] = qda(vcol(embfitm.T, 1:a), y, weights; kwargs...)
    end
    fitm = (embfitm = embfitm, dafitm = dafitm)
    Plsprobda(fitm, res.lev, ni, par)
end



