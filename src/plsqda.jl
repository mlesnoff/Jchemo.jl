"""
    plsqda(; kwargs...)
    plsqda(X, y; kwargs...)
    plsqda(X, y, weights::Weight; kwargs...)
PLS-QDA (with continuum).
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs) to compute.
    Must be >= 1.
* `prior` : Type of prior probabilities for class 
    membership. Possible values are: `:unif` (uniform), 
    `:prop` (proportional).
* `alpha` : Scalar (∈ [0, 1]) defining the continuum
    between QDA (`alpha = 0`) and LDA (`alpha = 1`).
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

QDA on PLS latent variables.The training variable `y` 
(univariate class membership) is transformed to a dummy table 
(Ydummy) containing nlev columns, where nlev is the number of 
classes present in `y`. Each column of Ydummy is a dummy (0/1) 
variable. Then, a PLSR2 (i.e. multivariate) is run on 
{`X`, Ydummy}, returning a score matrix `T`. Finally, a QDA 
(possibly with continuum) is done on {`T`, `y`}. 

See function `plslda` for examples.
""" 
function plsqda(X, y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    plsqda(X, y, weights; kwargs...)
end

function plsqda(X, y, weights::Weight; kwargs...)
    par = recovkwargs(Par, kwargs)
    @assert par.nlv >= 1 "Argument 'nlv' must be in >= 1"   
    res = dummy(y)
    ni = tab(y).vals
    fmpls = plskern(X, res.Y, weights;
        kwargs...)
    fmda = list(Qda, par.nlv)
    @inbounds for i = 1:par.nlv
        fmda[i] = qda(vcol(fmpls.T, 1:i), y, weights; kwargs...)
    end
    fm = (fmpls = fmpls, fmda = fmda)
    Plslda(fm, res.lev, ni)
end


