"""
    splskdeda(X, y, weights = ones(nro(X)); nlv, 
        msparse = :soft, delta = 0, nvar = nco(X), 
        prior = :unif, h = nothing, a = 1, scal::Bool = false)
Sparse PLS-KDE-DA.
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`). 
* `nlv` : Nb. latent variables (LVs) to compute.
* `msparse`: Method used for the thresholding. Possible values
    are :soft (default), :mix or :hard. See thereafter.
* `delta` : Range for the thresholding (see function `soft`)
    on the loadings standardized to their maximal absolute value.
    Must ∈ [0, 1]. Only used if `msparse = :soft.
* `nvar` : Nb. variables (`X`-columns) selected for each 
    LV. Can be a single integer (same nb. variables
    for each LV), or a vector of length `nlv`.
    Only used if `msparse = :mix` or `msparse = :hard`. 
* `prior` : Type of prior probabilities for class membership.
    Possible values are: :unif (uniform; default), :prop (proportional).
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

Same as function `plskdeda` (PLS-KDE-DA) except that sparse PLSR (function 
`splskern`) is run on the Y-dummy table instead of a PLSR (function `plskern`). 

See function `splskern` and `?plskdeda.

See function `splslda` for examples.
""" 
function splskdeda(X, y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    splskdeda(X, y, weights; 
        kwargs...)
end

function splskdeda(X, y, weights::Weight; 
        kwargs...)
    par = recovkwargs(Par, kwargs)
    @assert par.nlv >= 1 "Argument 'nlv' must be in >= 1"   
    res = dummy(y)
    ni = tab(y).vals
    fmpls = splskern(X, res.Y, weights; 
        kwargs...)
    fmda = list(Kdeda, par.nlv)
    @inbounds for i = 1:par.nlv
        fmda[i] = kdeda(vcol(fmpls.T, 1:i), y; 
            kwargs...)
    end
    fm = (fmpls = fmpls, fmda = fmda)
    Plslda(fm, res.lev, ni)
end


