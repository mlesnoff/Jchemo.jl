struct Occsdod
    d::DataFrame
    fm_sd
    fm_od
end

"""
    occsdod(object::Union{Pca, Plsr}, X; 
        nlv_sd = nothing, nlv_od = nothing, 
        typc = "mad", cri = 3, alpha = .05)
One-class classification using a compromise between PCA/PLS score (SD) and orthogonal (OD) distances.

* `object` : The fitted PCA/PLS model.
* `nlv_sd` : Nb. components (PCs or LVs) to consider for SD. If nothing, 
    it is the maximum nb. of components of the fitted model.
* `nlv_od` : Nb. components (PCs or LVs) to consider for OD. If nothing, 
    it is the maximum nb. of components of the fitted model.
* `typc` : Type of cutoff ("mad" or "q"). See below.
* `cri` : When `typc = "mad"`, constant used for computing the 
    cutoff detecting extreme values.
* `alpha` : When `typc = "q"`, risk-I level used for computing the cutoff 
    detecting extreme values.

The function computes a compromise between the score distance (SD) and the
orthogonal distance (OD). The compromise is: dstand = sqrt(sd_stand * od_stand).

See `?occsd` and `?occod` for details.
""" 
function occsdod(object::Union{Pca, Plsr}, X; 
        nlv_sd = nothing, nlv_od = nothing, 
        typc = "mad", cri = 3, alpha = .05)
    fm_sd = occsd(object; nlv = nlv_sd,
        typc = typc, cri = cri, alpha = alpha)
    fm_od = occod(object, X; nlv = nlv_sd,
        typc = typc, cri = cri, alpha = alpha)
    sd = fm_sd.d
    od = fm_od.d
    z = sqrt.(sd.dstand .* od.dstand)
    nam = string.("sd_", names(sd))
    rename!(sd, nam)
    nam = string.("od_", names(od))
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
    nam = string.("sd_", names(sd))
    rename!(sd, nam)
    nam = string.("od_", names(od))
    rename!(od, nam)
    d = hcat(sd, od)
    d.dstand = dstand
    pred = reshape(Int64.(dstand .> 1), m, 1)
    (pred = pred, d)
end



