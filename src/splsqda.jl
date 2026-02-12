"""
    splsqda(; kwargs...)
    splsqda(X, y; kwargs...)
    splsqda(X, y, weights::Weight; kwargs...)
Sparse PLS-QDA (with continuum).
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g., function `mweight`). 
Keyword arguments: 
* `nlv` : Nb. latent variables (LVs) to compute.
* `meth` : Method used for the sparse thresholding. Possible values are: `:soft`, `:hard`. See thereafter.
* `nvar` : Nb. variables (`X`-columns) selected for each LV. Can be a single integer (i.e. same nb. 
    of variables for each LV), or a vector of length `nlv`.   
* `prior` : Type of prior probabilities for class membership. Possible values are: `:prop` (proportionnal), 
    `:unif` (uniform), or a vector (of length equal to the number of classes) giving the prior weight for each class 
    (in case of vector, it must be sorted in the same order as `mlev(y)`).
* `alpha` : Scalar (∈ [0, 1]) defining the continuum between QDA (`alpha = 0`) and LDA (`alpha = 1`).
* `tol` : Only when q > 1; tolerance used in function `snipals_shen`. 
* `maxit` : Only when q > 1; maximum nb. of iterations used in function `snipals_shen`.    
* `scal` : Boolean. If `true`, each column of `X` and `Y` is scaled by its uncorrected standard deviation.    

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
    priors = aggsumv(weights.w, vec(y)).val  # output not used, only for information
    fitm_emb = splsr(X, res.Y, weights; kwargs...)
    fitm_da = list(Qda, par.nlv)
    @inbounds for i = 1:par.nlv
        fitm_da[i] = qda(vcol(fitm_emb.T, 1:i), y, weights; kwargs...)
    end
    Plsprobda(fitm_emb, fitm_da, ni, priors, res.lev, par) 
end


