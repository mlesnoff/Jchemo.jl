"""
    occsdod(object::Union{Pca, Plsr}, X; 
        nlv_sd = nothing, nlv_od = nothing, 
        typc = :mad, cri = 3, alpha = .025, kwargs...)
One-class classification using a compromise between PCA/PLS score (SD) and 
    orthogonal (OD) distances.

* `object` : The model (e.g. PCA) that was fitted on the training data,
    assumed to represent the training class.
* `X` : X-data (training) that were used to fit the model.
* `nlv_sd` : Nb. components (PCs or LVs) to consider for SD. If nothing, 
    it is the maximum nb. of components of the fitted model.
* `nlv_od` : Nb. components (PCs or LVs) to consider for OD. If nothing, 
    it is the maximum nb. of components of the fitted model.
* `typc` : Type of cutoff (:mad or :q). See Thereafter.
* `cri` : When `typc = :mad`, a constant. See thereafter.
* `alpha` : When `typc = :q`, a risk-I level. See thereafter.
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
        typc = :mad, cri = 3, alpha = .025, kwargs...)
    fm_sd = occsd(object; nlv = nlv_sd,
        typc = typc, cri = cri, alpha = alpha)
    fm_od = occod(object, X; nlv = nlv_od,
        typc = typc, cri = cri, alpha = alpha)
    sd = fm_sd.d
    od = fm_od.d
    z = sqrt.(sd.dstand .* od.dstand)
    nam = string.(names(sd), "_sd")
    rename!(sd, nam)
    nam = string.(names(od), "_od")
    rename!(od, nam)
    d = hcat(sd, od)
    d.dstand = z
    Occsdod(d, fm_sd, fm_od)
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
    sd = predict(object.fm_sd, X).d
    od = predict(object.fm_od, X).d
    dstand = sqrt.(sd.dstand .* od.dstand)
    nam = string.(names(sd), "_sd")
    rename!(sd, nam)
    nam = string.(names(od), "_od")
    rename!(od, nam)
    d = hcat(sd, od)
    d.dstand = dstand
    pred = reshape(Int64.(dstand .> 1), m, 1)
    (pred = pred, d)
end



