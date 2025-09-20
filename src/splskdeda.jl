"""
    splskdeda(; kwargs...)
    splskdeda(X, y; kwargs...)
    splskdeda(X, y, weights::Weight; kwargs...)
Sparse PLS-KDE-DA.
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g. function `mweight`). 
Keyword arguments: 
* Same as function `splsr`, and the following:         
* `prior` : Type of prior probabilities for class membership. Possible values are: `:prop` (proportionnal), 
    `:unif` (uniform), or a vector (of length equal to the number of classes) giving the prior weight for each class 
    (in case of vector, it must be sorted in the same order as `mlev(y)`).
* Eventua keyword arguments of function `dmkern` (bandwidth definition).

Same as function `plskdeda` (PLS-KDEDA) except that a sparse PLSR (function `splsr`), instead of a PLSR, 
is run on the Y-dummy table. 

See function `splslda` for examples.
""" 
splskdeda(; kwargs...) = JchemoModel(splskdeda, nothing, kwargs)

function splskdeda(X, y; kwargs...)
    par = recovkw(ParSplskdeda, kwargs).par
    Q = eltype(X[1, 1])
    weights = mweightcla(Q, y; prior = par.prior)
    splskdeda(X, y, weights; kwargs...)
end

function splskdeda(X, y, weights::Weight; kwargs...)
    par = recovkw(ParSplskdeda, kwargs).par
    @assert par.nlv >= 1 "Argument 'nlv' must be in >= 1"   
    res = dummy(y)
    ni = tab(y).vals
    embfitm = splsr(X, res.Y, weights; kwargs...)
    dafitm = list(Kdeda, par.nlv)
    @inbounds for i = 1:par.nlv
        dafitm[i] = kdeda(vcol(embfitm.T, 1:i), y; kwargs...)
    end
    fitm = (embfitm = embfitm, dafitm = dafitm)
    Plsprobda(fitm, res.lev, ni, par)
end


