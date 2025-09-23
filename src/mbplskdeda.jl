"""
    mbplskdeda(; kwargs...)
    mbplskdeda(Xbl, y; kwargs...)
    mbplskdeda(Xbl, y, weights::Weight; kwargs...)
Multiblock PLS-KDEDA.
* `Xbl` : List of blocks (vector of matrices) of X-data. Typically, output of function `mblock` from data (n, p).  
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs; = scores) to compute.
* `bscal` : Type of block scaling. See function `blockscal` for possible values.
* `prior` : Type of prior probabilities for class membership. Possible values are: `:prop` (proportionnal), 
    `:unif` (uniform), or a vector (of length equal to the number of classes) giving the prior weight for each class 
    (in case of vector, it must be sorted in the same order as `mlev(y)`).
* Keyword arguments of function `dmkern` (bandwidth definition) can also be specified here.
* `scal` : Boolean. If `true`, each column of blocks in `Xbl` and Ydummy is scaled by its uncorrected standard deviation 
    (before the block scaling) in the MBPLS computation.

Same as function `mbplsqda` except that the class densities are estimated from `dmkern` instead of `dmnorm`.

See function `mbplslda` for examples.

""" 
mbplskdeda(; kwargs...) = JchemoModel(mbplskdeda, nothing, kwargs)

function mbplskdeda(Xbl, y; kwargs...)
    par = recovkw(ParMbplskdeda, kwargs).par
    Q = eltype(Xbl[1][1, 1])
    weights = mweightcla(Q, y; prior = par.prior)
    mbplskdeda(Xbl, y, weights; kwargs...)
end

function mbplskdeda(Xbl, y, weights::Weight; kwargs...)
    par = recovkw(ParMbplskdeda, kwargs).par
    @assert par.nlv >= 1 "Argument 'nlv' must be in >= 1"   
    res = dummy(y)
    ni = tab(y).vals
    fitm_emb = mbplsr(Xbl, res.Y, weights; kwargs...)
    fitm_da = list(Kdeda, par.nlv)
    @inbounds for i = 1:par.nlv
        fitm_da[i] = kdeda(vcol(fitm_emb.T, 1:i), y; kwargs...)
    end
    Mbplsprobda(fitm_emb, fitm_da, res.lev, ni, par)
end

