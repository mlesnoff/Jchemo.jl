"""
    kplskdeda(; kwargs...)
    kplskdeda(X, y; kwargs...)
    kplskdeda(X::Matrix{Q}, y::Vector{String}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: Float
KPLS-KDEDA.
* `X` : X-data (n, p).
* `y` : Univariate class membership (n). Must be a `Vector{String}`.
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs) to compute. Must be >= 1. 
* `kern` : Type of kernel used to compute the Gram matrices. Possible values are: `:krbf`, `:kpol`. See respective 
    functions `krbf` and `kpol` for their keyword arguments.
* `prior` : Type of prior probabilities for class membership. Possible values are: `:prop` (proportionnal), 
    `:unif` (uniform), or a vector (of length equal to the number of classes) giving the prior weight for each class 
    (in case of vector, it must be sorted in the same order as `mlev(y)`).
* Eventual keyword arguments of function `dmkern` (bandwidth definition).
* `scal` : Boolean. If `true`, each column of `X` and Ydummy is scaled by its uncorrected standard deviation
    in the PLS computation.

Same as function `plskdeda` (PLS-KDEDA) except that a kernel PLSR (function `kplsr`), instead of a  PLSR (function `plskern`), 
is run on the Y-dummy table. 

See function `kplslda` for examples.
""" 
kplskdeda(; kwargs...) = JchemoModel(kplskdeda, nothing, kwargs)

function kplskdeda(X, y; kwargs...)
    par = recovkw(ParKplskdeda{Q}, kwargs).par
    Q = eltype(X[1, 1])
    weights = pweightcla(Q, y; prior = par.prior)
    kplskdeda(X, y, weights; kwargs...)
end

function kplskdeda(X::Matrix{Q}, y::Vector{String}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: Float
    par = recovkw(ParKplskdeda{Q}, kwargs).par
    @assert par.nlv >= 1 "Argument 'nlv' must be in >= 1"   
    res = dummy(Q, y)
    ni = tab(y).vals
    priors = aggsumv(weights.values, vec(y)).val  # output not used, only for information
    fitm_emb = kplsr(X, res.Y, weights; kwargs...)
    par.nlv = fitm_emb.par.nlv
    fitm_da = list(Kdeda, par.nlv)
    @inbounds for i in eachindex(fitm_da)
        fitm_da[i] = kdeda(vcol(fitm_emb.T, 1:i), y, weights; kwargs...)
    end
    Plsprobda(fitm_emb, fitm_da, ni, priors, res.lev, par) 
end



