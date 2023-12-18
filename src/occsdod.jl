"""
    occsdod(object::Union{Pca, Plsr}, X; 
        nlv_sd = nothing, nlv_od = nothing, 
        mcut = :mad, cri = 3, risk = .025, kwargs...)
One-class classification using a compromise between PCA/PLS score (SD) and 
    orthogonal (OD) distances.

* `object` : The model (e.g. PCA) that was fitted on the training data,
    assumed to represent the training class.
* `X` : X-data (training) that were used to fit the model.
* `nlv_sd` : Nb. components (PCs or LVs) to consider for SD. If nothing, 
    it is the maximum nb. of components of the fitted model.
* `nlv_od` : Nb. components (PCs or LVs) to consider for OD. If nothing, 
    it is the maximum nb. of components of the fitted model.
* `mcut` : Type of cutoff (:mad or :q). See Thereafter.
* `cri` : When `mcut = :mad`, a constant. See thereafter.
* `risk` : When `mcut = :q`, a risk-I level. See thereafter.
* `kwargs` : Optional arguments to pass in function `kde` of 
    KernelDensity.jl (see function `kde1`).

In this method, the outlierness `d` of a given observation
is a compromise between the score distance (SD) and the
orthogonal distance (OD). The compromise is computed from the 
standardized distances by: 
* `dstand` = sqrt(`dstand_sd` * `dstand_od`).

See `?occsd` for details on outputs, and examples.
""" 
function occsdod(object::Union{Pca, Plsr}, X; 
        nlv_sd = nothing, nlv_od = nothing, 
        mcut = :mad, cri = 3, risk = .025, kwargs...)
    fmsd = occsd(object; nlv = nlv_sd,
        mcut = mcut, cri = cri, risk = risk)
    fmod = occod(object, X; nlv = nlv_od,
        mcut = mcut, cri = cri, risk = risk)
    sd = fmsd.d
    od = fmod.d
    z = sqrt.(sd.dstand .* od.dstand)
    nam = string.(names(sd), "_sd")
    rename!(sd, nam)
    nam = string.(names(od), "_od")
    rename!(od, nam)
    d = hcat(sd, od)
    d.dstand = z
    Occsdod(d, fmsd, fmod)
end

"""
    predict(object::Occsdod, X)
Compute predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Occsdod, X)
    X = ensure_mat(X)
    m = nro(X)
    sd = predict(object.fmsd, X).d
    od = predict(object.fmod, X).d
    dstand = sqrt.(sd.dstand .* od.dstand)
    nam = string.(names(sd), "_sd")
    rename!(sd, nam)
    nam = string.(names(od), "_od")
    rename!(od, nam)
    d = hcat(sd, od)
    d.dstand = dstand
    pred = [if d.dstand[i] <= 1 ; "in" else "out" ; 
        end ; for i = 1:m]
    pred = reshape(pred, m, 1)
    (pred = pred, d)
end



