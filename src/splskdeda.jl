"""
    splskdeda(; kwargs...)
    splskdeda(X, y; kwargs...)
    splskdeda(X, y, weights::Weight; kwargs...)
Sparse PLS-KDE-DA.
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g. function `mweight`). 
Keyword arguments: 
* `nlv` : Nb. latent variables (LVs) to compute.
* `meth` : Method used for the sparse thresholding. Possible values are: `:soft`, `:hard`. See thereafter.
* `nvar` : Nb. variables (`X`-columns) selected for each LV. Can be a single integer (i.e. same nb. 
    of variables for each LV), or a vector of length `nlv`.   
* `prior` : Type of prior probabilities for class membership. Possible values are: `:prop` (proportionnal), 
    `:unif` (uniform), or a vector (of length equal to the number of classes) giving the prior weight for each class 
    (in case of vector, it must be sorted in the same order as `mlev(y)`).
* Eventual keyword arguments of function `dmkern` for bandwidth definition.
* `tol` : Only when q > 1; tolerance used in function `snipals_shen`. 
* `maxit` : Only when q > 1; maximum nb. of iterations used in function `snipals_shen`.    
* `scal` : Boolean. If `true`, each column of `X` and `Y` is scaled by its uncorrected standard deviation. 

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
    fitm_emb = splsr(X, res.Y, weights; kwargs...)
    fitm_da = list(Kdeda, par.nlv)
    @inbounds for i = 1:par.nlv
        fitm_da[i] = kdeda(vcol(fitm_emb.T, 1:i), y; kwargs...)
    end
    Plsprobda(fitm_emb, fitm_da, res.lev, ni, par) 
end


