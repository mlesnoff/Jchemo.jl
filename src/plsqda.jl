"""
    plsqda(X, y, weights = ones(nro(X)); nlv, 
        alpha = 0, prior = :unif, scal::Bool = false)
PLS-QDA (with continuum).
* `X` : X-data.
* `y` : y-data (class membership).
* `weights` : Weights of the observations. 
    Internally normalized to sum to 1. 
* `nlv` : Nb. latent variables (LVs) to compute.
* `alpha` : Scalar (∈ [0, 1]) defining the continuum
    between QDA (`alpha = 0`; default) and LDA (`alpha = 1`).
* `prior` : Type of prior probabilities for class membership.
    Possible values are: :unif (uniform; default), :prop (proportional).
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

QDA on PLS latent variables. 

The training variable `y` (univariate class membership) is transformed
to a dummy table (Ydummy) containing nlev columns, where nlev is the number 
of classes present in `y`. Each column of Ydummy is a dummy variable (0/1). 
Then, a PLS2 is implemented on `X` and Ydummy, 
returning `nlv` latent variables (LVs). Finally, a QDA is run on these LVs and `y`.

See `?plslda` for examples.
""" 
function plsqda(X, y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    plsqda(X, y, weights; 
        kwargs...)
end

function plsqda(X, y, weights::Weight; 
        kwargs...)
    par = recovkwargs(Par, kwargs)
    res = dummy(y)
    ni = tab(y).vals
    fmpls = plskern(X, res.Y, weights;
        kwargs...)
    fmda = list(par.nlv, Qda)
    @inbounds for i = 1:par.nlv
        fmda[i] = qda(vcol(fmpls.T, 1:i), y, weights; 
            kwargs...)
    end
    fm = (fmpls = fmpls, fmda = fmda)
    Plslda(fm, res.lev, ni)
end


