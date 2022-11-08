"""
    plsqda(X, y, weights = ones(nro(X)); nlv, 
        prior = "unif", scal = false)
QDA on PLS latent variables.
* `X` : X-data.
* `y` : y-data (class membership).
* `weights` : Weights of the observations.
* `nlv` : Nb. latent variables (LVs) to compute.
* `prior` : Type of prior probabilities for class membership.
    Posible values are: "unif" (uniform), "prop" (proportional).
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

The training variable `y` (univariate class membership) is transformed
to a dummy table (Ydummy) containing nlev columns, where nlev is the number 
of classes present in `y`. Each column of Ydummy is a dummy variable (0/1). 
Then, a PLS2 is implemented on `X` and Ydummy, 
returning `nlv` latent variables (LVs). Finally, a QDA is run on these LVs and `y`.

See `?plslda` for examples.
""" 
function plsqda(X, y, weights = ones(nro(X)); nlv, 
        prior = "unif", scal = false)
    z = dummy(y)
    fm_pls = plskern(X, z.Y, weights; nlv = nlv, scal = scal)
    fm_da = list(nlv)
    for i = 1:nlv
        fm_da[i] = qda(vcol(fm_pls.T, 1:i), y; prior = prior)
    end
    fm = (fm_pls = fm_pls, fm_da = fm_da)
    PlsLda(fm, z.lev, z.ni)
end


