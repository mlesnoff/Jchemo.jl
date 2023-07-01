""" 
    plsqdaavg(X, y, weights = ones(nro(X)); nlv,
        scal = false)
Averaging of PLS-QDA models with different numbers of 
    latent variables (LVs).
* `X` : X-data.
* `y` : y-data (class membership).
* weights : Weights of the observations.
* `nlv` : A character string such as "5:20" defining the range of the numbers of LVs 
    to consider ("5:20": the predictions of models with nb LVS = 5, 6, ..., 20 
    are averaged). Syntax such as "10" is also allowed ("10": correponds to
    the single model with 10 LVs).
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

Ensemblist method where the predictions are calculated by "averaging" 
the predictions of a set of models built with different numbers of LVs.

For instance, if argument `nlv` is set to `nlv = "5:10"`, the prediction for 
a new observation is the most occurent class within the predictions 
returned by the models with 5 LVS, 6 LVs, ... 10 LVs, respectively.

See `?plsldaavg` for examples.
""" 
function plsqdaavg(X, y, weights = ones(nro(X)); nlv,
        scal = false)
    n = nro(X)
    p = nco(X)
    nlv = eval(Meta.parse(nlv))
    nlvmax = maximum(nlv)
    nlv = (max(minimum(nlv), 0):min(nlvmax, n, p))
    w = ones(nlvmax + 1)
    # Uniform weights for the models
    w_mod = mweight(w[collect(nlv) .+ 1])
    # End   
    fm = plsqda(X, y, weights; nlv = nlvmax,
        scal = scal)
    Plsdaavg(fm, nlv, w_mod, fm.lev, fm.ni)
end

