"""
    splsqda(; kwargs...)
    splsqda(X, y; kwargs...)
    splsqda(X, y, weights::Weight; kwargs...)
Sparse PLS-QDA (with continuum).
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g. function `mweight`). 
Keyword arguments: 
* Same as function `splsr`, and the following:         
* `prior` : Type of prior probabilities for class membership. Possible values are: `:prop` (proportionnal), 
    `:unif` (uniform), or a vector (of length equal to the number of classes) giving the prior weight for each class 
    (in case of vector, it must be sorted in the same order as `mlev(y)`).         
* `alpha` : Scalar (∈ [0, 1]) defining the continuum between QDA (`alpha = 0`) and LDA (`alpha = 1`).

Same as function `plsqda` (PLSR-QDA) except that a sparse PLSR (function `splsr`), instead of a PLSR, 
is run on the Y-dummy table.

See function `splslda` for examples.
""" 
splsqda(; kwargs...) = JchemoModel(splsqda, nothing, kwargs)

function splsqda(X, y; kwargs...)
    par = recovkw(ParSplsqda, kwargs).par
    Q = eltype(X[1, 1])
    weights = mweightcla(Q, y; prior = par.prior)
    splsqda(X, y, weights; kwargs...)
end

function splsqda(X, y, weights::Weight; kwargs...)
    par = recovkw(ParSplsqda, kwargs).par
    @assert par.nlv >= 1 "Argument 'nlv' must be in >= 1"   
    res = dummy(y)
    ni = tab(y).vals
    embfitm = splsr(X, res.Y, weights; kwargs...)
    dafitm = list(Qda, par.nlv)
    @inbounds for i = 1:par.nlv
        dafitm[i] = qda(vcol(embfitm.T, 1:i), y, weights; kwargs...)
    end
    fitm = (embfitm = embfitm, dafitm = dafitm)
    Plsprobda(fitm, res.lev, ni, par)
end


