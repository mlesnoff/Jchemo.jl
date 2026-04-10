"""
    occdds(; kwargs...)
    occdds(object, X; kwargs...)
One-class classification using a consensus between PCA/PLS score and orthogonal distances (SD and OD).
* `fitm` : The preliminary model (e.g., object `fitm` returned by function `pcasvd`) that was fitted on 
    the training data assumed to represent the reference class.
* `X` : Training X-data (n, p), on which was fitted the model `fitm`.
Keyword arguments:
* `cut` : Type of cutoff. Possible values are: `:mad`, `:q`. See Thereafter.
* `cri` : When `cut` = `:mad`, a constant. See thereafter.
* `alpha` : When `cut` = `:q`, a alpha-I level. See thereafter.

In this method, outlierness `d` of a given observation is a consensus between the score distance (SD) and the
orthogonal distance (OD). The consensus is computed from the standardized distances by: 
* `dstand` = sqrt(`dstand_sd` * `dstand_od`).

See functions:
* `occsd` for details on the cutoff computation and the outputs,
* and `occod` for examples.
""" 
#occdds(; kwargs...) = JchemoModel(occdds, nothing, kwargs)

function occdds(fitm, X; fcentr = meanv, fscal = stdv, alpha = .05) 
    #par = recovkw(ParOcc, kwargs).par 
    sd = outsd(fitm).d
    od = outod(fitm, X).d
    ##
    ##
    d = sd.^2 
    mu = fcentr(d)
    sigma = fscal(d)
    g = sigma^2 / (2 * mu)
    nu = 2 * (mu / sigma)^2
    nu = max(1, round(Int, nu))
    cutoff = mu / nu * quantile(Chisq(nu), 1 - alpha)
    sd2 = (d = d, mu, sigma, g, nu, cutoff)
    #quantile(Chisq(nlv), 1 - alpha)
    ##
    d = od.^2 
    mu = fcentr(d)
    sigma = fscal(d)
    g = sigma^2 / (2 * mu)
    nu = 2 * (mu / sigma)^2
    nu = max(1, round(Int, nu))
    cutoff = mu / nu * quantile(Chisq(nu), 1 - alpha)
    od2 = (d = d, mu, sigma, g, nu, cutoff)
    ##
    d = sd2.nu / sd2.mu * sd2.d + od2.nu / od2.mu * od2.d 
    nu = sd2.nu + od2.nu
    cutoff = quantile(Chisq(nu), 1 - alpha)
    ## 
    (d = d, nu, cutoff, sd2, od2)
end

"""
    predict(object::Occsdod, X)
Compute predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict_occds(object, X)
    m = nro(X)
    sd = predict(object.fitmsd, X).d
    od = predict(object.fitmod, X).d
    nam = string.(names(sd), "_sd")
    rename!(sd, nam)
    nam = string.(names(od), "_od")
    rename!(od, nam)
    d = hcat(sd, od)
    d.dstand = [sqrt(sd.dstand_sd[i] * od.dstand_od[i]) for i in eachindex(sd.dstand_sd)]
    pred = [if d.dstand[i] <= 1 "in" else "out" end for i in eachindex(d.d)]
    pred = reshape(pred, m, 1)
    (pred = pred, d)
end

