"""
    occsdod(; kwargs...)
    occsdod(object, X; kwargs...)
One-class classification using a compromise between 
    PCA/PLS score (SD) and orthogonal (OD) distances.
* `fitm` : The preliminary model (e.g. PCA; object `fitm`) that was fitted 
    on the training data assumed to represent the training class.
* `X` : Training X-data (n, p), on which was fitted 
    the model `fitm`.
Keyword arguments:
* `mcut` : Type of cutoff. Possible values are: `:mad`, `:q`. 
    See Thereafter.
* `cri` : When `mcut` = `:mad`, a constant. See thereafter.
* `risk` : When `mcut` = `:q`, a risk-I level. See thereafter.

In this method, the outlierness `d` of a given observation
is a compromise between the score distance (SD) and the
orthogonal distance (OD). The compromise is computed from the 
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
    z = sqrt.(sd.dstand .* od.dstand)
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
    d.dstand = sqrt.(sd.dstand_sd .* od.dstand_od)
    pred = [if d.dstand[i] <= 1 "in" else "out" end for i = 1:m]
    pred = reshape(pred, m, 1)
    (pred = pred, d)
end

