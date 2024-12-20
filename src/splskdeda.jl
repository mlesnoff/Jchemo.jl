"""
    splskdeda(; kwargs...)
    splskdeda(X, y; kwargs...)
    splskdeda(X, y, weights::Weight; kwargs...)
Sparse PLS-KDE-DA.
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`). 
Keyword arguments: 
* `nlv` : Nb. latent variables (LVs) to compute.
    Must be >= 1.
* `meth` : Method used for the sparse thresholding. 
    Possible values are: `:soft2`, `:soft`, 
    `:hard`. See thereafter.
* `delta` : Only used if `meth = :soft2`. Range for the 
    thresholding on the loadings (after they are standardized 
    to their maximal absolute value). Must ∈ [0, 1].
    Higher is `delta`, stronger is the thresholding. 
* `nvar` : Only used if `meth = :soft` or `meth = :hard`.
    Nb. variables (`X`-columns) selected for each principal
    component (PC). Can be a single integer (i.e. same nb. 
    of variables for each PC), or a vector of length `nlv`.   
* `prior` : Type of prior probabilities for class 
    membership. Possible values are: `:unif` (uniform), 
    `:prop` (proportional), or a vector (of length equal to 
    the number of classes) giving the prior weight for each class 
    (in case of vector, it must be sorted in the same order as `mlev(y)`).
* Keyword arguments of function `dmkern` (bandwidth 
    definition) can also be specified here.
* `scal` : Boolean. If `true`, each column of `X` 
    and Ydummy is scaled by its uncorrected standard deviation
    in the PLS computation.

Same as function `plskdeda` (PLS-LDA) except that 
a sparse PLSR (function `splsr`), instead of a 
PLSR (function `plskern`), is run on the Y-dummy table. 

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


