"""
    splskdeda(X, y, weights = ones(nro(X)); nlv, 
        meth = :soft, delta = 0, nvar = nco(X), 
        prior = :unif, h = nothing, a = 1, scal::Bool = false)
Sparse PLS-KDE-DA.
* `X` : X-data.
* `y` : y-data (class membership).
* `weights` : Weights of the observations. Internally normalized to sum to 1. 
* `nlv` : Nb. latent variables (LVs) to compute.
* `meth`: Method used for the thresholding. Possible values
    are :soft (default), :mix or :hard. See thereafter.
* `delta` : Range for the thresholding (see function `soft`)
    on the loadings standardized to their maximal absolute value.
    Must ∈ [0, 1]. Only used if `meth = :soft.
* `nvar` : Nb. variables (`X`-columns) selected for each 
    LV. Can be a single integer (same nb. variables
    for each LV), or a vector of length `nlv`.
    Only used if `meth = :mix` or `meth = :hard`. 
* `prior` : Type of prior probabilities for class membership.
    Possible values are: :unif (uniform; default), :prop (proportional).
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

Same as function `plskdeda` (PLS-KDE-DA) except that sparse PLSR (function 
`splskern`) is run on the Y-dummy table instead of a PLSR (function `plskern`). 

See `?splskern` and `?plskdeda.

See `?splslda` for examples.
""" 
function splskdeda(X, y, weights = ones(nro(X)); nlv, 
        meth = :soft, delta = 0, nvar = nco(X),
        prior = :unif, h = nothing, a = 1, scal::Bool = false)
    res = dummy(y)
    ni = tab(y).vals
    fmpls = splskern(X, res.Y, weights; nlv = nlv, 
        meth = meth, delta = delta, nvar = nvar, 
        scal = scal)
    fmda = list(nlv)
    @inbounds for i = 1:nlv
        fmda[i] = kdeda(vcol(fmpls.T, 1:i), y; prior = prior,
            h = h, a = a)
    end
    fm = (fmpls = fmpls, fmda = fmda)
    Plslda(fm, res.lev, ni)
end


