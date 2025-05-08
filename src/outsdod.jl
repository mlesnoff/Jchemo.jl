"""
    outsdod(fitm, X; cut = :mad, cri = 3, risk = .025)
Compute outlierness from PCA/PLS score and orthogonal distances (SD and OD).
* `fitm` : The preliminary model (e.g. object `fitm` returned by function `pcasvd`) that was fitted on 
    the data.
* `X` : X-data (n, p) on which was fitted the model `fitm`.
Keyword arguments:
* `cut` : Type of cutoff to standardize SD and OD. Possible values are: `:mad`, `:q`. See Thereafter.
* `cri` : When `cut` = `:mad`, a constant. See thereafter.
* `risk` : When `cut` = `:q`, a risk-I level. See thereafter.

In this method, the outlierness `d` of a given observation is a consensus between the standardized score and
orthogonal distances. The returned consensus is computed by: 
* `d` = sqrt(SD_stand * OD_stand)
where:
* SD_stand = SD / cutoff_SD
* OD_stand = OD / cutoff_OD

""" 
function outsdod(fitm, X; cut = :mad, cri = 3, risk = .025)
    d_sd = outsd(fitm)
    d_od = outod(fitm, X)
    dstand_sd = d_sd / cutoff_sd
    dstand_od = d_od / cutoff_od
    d = [sqrt(d_sd[i] * d_od[i]) for i in eachindex(sd.dstand)]
    (d = d,)
end



