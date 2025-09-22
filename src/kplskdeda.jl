"""
    kplskdeda(; kwargs...)
    kplskdeda(X, y; kwargs...)
    kplskdeda(X, y, weights::Weight; kwargs...)
KPLS-KDEDA.
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

Same as function `plskdeda` (PLS-KDEDA) except that a kernel PLSR (function `kplsr`), instead of a  PLSR (function `plskern`), 
is run on the Y-dummy table. 

See function `kplslda` for examples.
""" 
kplskdeda(; kwargs...) = JchemoModel(kplskdeda, nothing, kwargs)

function kplskdeda(X, y; kwargs...)
    par = recovkw(ParKplskdeda, kwargs).par
    Q = eltype(X[1, 1])
    weights = mweightcla(Q, y; prior = par.prior)
    kplskdeda(X, y, weights; kwargs...)
end

function kplskdeda(X, y, weights::Weight; kwargs...)
    par = recovkw(ParKplskdeda, kwargs).par
    @assert par.nlv >= 1 "Argument 'nlv' must be in >= 1"   
    res = dummy(y)
    ni = tab(y).vals
    fitm_emb = kplsr(X, res.Y, weights; kwargs...)
    fitm_da = list(Kdeda, par.nlv)
    @inbounds for a = 1:par.nlv
        fitm_da[a] = kdeda(vcol(fitm_emb.T, 1:a), y; kwargs...)
    end
    Plsprobda(fitm_emb, fitm_da, res.lev, ni, par) 
end



