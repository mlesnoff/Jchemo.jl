"""
    dkplskdeda(; kwargs...)
    dkplskdeda(X, y; kwargs...)
    dkplskdeda(X, y, weights::Weight; kwargs...)
DKPLS-KDEDA.
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs) to compute. Must be >= 1.
* `kern` : Type of kernel used to compute the Gram matrices. Possible values are: `:krbf`, `:kpol`. See respective 
    functions `krbf` and `kpol` for their keyword arguments.
* `prior` : Type of prior probabilities for class membership. Possible values are: `:prop` (proportionnal), 
    `:unif` (uniform), or a vector (of length equal to the number of classes) giving the prior weight for each class 
    (in case of vector, it must be sorted in the same order as `mlev(y)`).
* Keyword arguments of function `dmkern` (bandwidth 
    definition) can also be specified here.
* `scal` : Boolean. If `true`, each column of `X` and Ydummy is scaled by its uncorrected standard deviation
    in the PLS computation.

Same as function `plskdeda` (PLS-KDEDA) except that a direct kernel PLSR (function `dkplsr`), instead of a PLSR 
(function `plskern`), is run on the Y-dummy table. 

See function `dkplslda` for examples.
""" 
dkplskdeda(; kwargs...) = JchemoModel(dkplskdeda, nothing, kwargs)

function dkplskdeda(X, y; kwargs...)
    par = recovkw(ParKplskdeda, kwargs).par
    Q = eltype(X[1, 1])
    weights = mweightcla(Q, y; prior = par.prior)
    dkplskdeda(X, y, weights; kwargs...)
end

function dkplskdeda(X, y, weights::Weight; kwargs...)
    par = recovkw(ParKplskdeda, kwargs).par
    @assert par.nlv >= 1 "Argument 'nlv' must be in >= 1"   
    res = dummy(y)
    ni = tab(y).vals
    embfitm = dkplsr(X, res.Y, weights; kwargs...)
    dafitm = list(Kdeda, par.nlv)
    @inbounds for i = 1:par.nlv
        dafitm[i] = kdeda(vcol(embfitm.T, 1:i), y; kwargs...)
    end
    fitm = (embfitm = embfitm, dafitm = dafitm)
    Plsprobda(fitm, res.lev, ni, par)
end



