"""
    occsdod(; kwargs...)
    occsdod(object, X; kwargs...)
One-class classification using a consensus between 
    PCA/PLS score and orthogonal (SD and OD) distances.
* `fitm` : The preliminary model (e.g. object `fitm` returned by function 
    `pcasvd`) that was fitted on the training data assumed to represent 
    the training class.
* `X` : Training X-data (n, p), on which was fitted the model `fitm`.
Keyword arguments:
* `cut` : Type of cutoff. Possible values are: `:mad`, `:q`. 
    See Thereafter.
* `cri` : When `cut` = `:mad`, a constant. See thereafter.
* `risk` : When `cut` = `:q`, a risk-I level. See thereafter.

In this method, the outlierness `d` of a given observation
is a consensus between the score distance (SD) and the
orthogonal distance (OD). The consensus is computed from the 
standardized distances by: 
* `dstand` = sqrt(`dstand_sd` * `dstand_od`).

See functions:
* `occsd` for details of the outputs,
* and `occod` for examples.
""" 
occsdod(; kwargs...) = JchemoModel(occsdod, nothing, kwargs)

function occsdod(fitm, X; kwargs...) 
    par = recovkw(ParOcc, kwargs).par 
    fitmsd = occsd(fitm; kwargs...)
    fitmod = occod(fitm, X; kwargs...)
    sd = fitmsd.d
    od = fitmod.d
    z = [sqrt(sd.dstand[i] * od.dstand[i]) for i in eachindex(sd.dstand)]
    nam = string.(names(sd), "_sd")
    rename!(sd, nam)
    nam = string.(names(od), "_od")
    rename!(od, nam)
    d = hcat(sd, od)
    d.dstand = z
    Occsdod(d, fitmsd, fitmod, par)
end

"""
    predict(object::Occsdod, X)
Compute predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Occsdod, X)
    m = nro(X)
    sd = predict(object.fitmsd, X).d
    od = predict(object.fitmod, X).d
    nam = string.(names(sd), "_sd")
    rename!(sd, nam)
    nam = string.(names(od), "_od")
    rename!(od, nam)
    d = hcat(sd, od)
    d.dstand = [sqrt(sd.dstand_sd[i] * od.dstand_od[i]) for i in eachindex(sd.dstand_sd)]
    pred = [if d.dstand[i] <= 1 "in" else "out" end for i = 1:m]
    pred = reshape(pred, m, 1)
    (pred = pred, d)
end

