"""
    plsqda(X, y, weights = ones(nro(X)); nlv, 
        alpha = 0, prior = "unif", scal = false)
PLS-QDA (with continuum towards PLS-LDA).
* `X` : X-data.
* `y` : y-data (class membership).
* `weights` : Weights of the observations. 
    Internally normalized to sum to 1. 
* `nlv` : Nb. latent variables (LVs) to compute.
* `alpha` : Scalar (∈ [0, 1]) defining the continuum
    between QDA (`alpha = 0`; default) and LDA (`alpha = 1`).
* `prior` : Type of prior probabilities for class membership.
    Posible values are: "unif" (uniform), "prop" (proportional).
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
function plsqda(X, y, weights = ones(nro(X)); nlv, 
        prior = "unif", scal = false)
    res = dummy(y)
    ni = tab(y).vals
    fm_pls = plskern(X, res.Y, weights; nlv = nlv, scal = scal)
    fm_da = list(nlv)
    @inbounds for i = 1:nlv
        fm_da[i] = qda(vcol(fm_pls.T, 1:i), y; prior = prior)
    end
    fm = (fm_pls = fm_pls, fm_da = fm_da)
    Plslda(fm, res.lev, ni)
end


