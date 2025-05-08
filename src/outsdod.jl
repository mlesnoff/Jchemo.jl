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

The cutoff is computed with non-parametric heuristics. Noting [d] the SD- or OD-vector:
* If `cut` = `:mad`, then `cutoff` = MED([d]) + `cri` * MAD([d]). 
* If `cut` = `:q`, then `cutoff` is estimated from the empirical cumulative density function 
  computed on [d], for a given risk-I (`risk`).

""" 
function outsdod(fitm, X; cut = :mad, cri = 3, risk = .025)
    @assert in(cut, [:mad, :q]) "Argument 'cut' must be :mad or :q."
    @assert 0 <= risk <= 1 "Argument 'risk' must ∈ [0, 1]."
    d_sd = outsd(fitm)
    d_od = outod(fitm, X)
    if cut == :mad
        cutoff_sd = median(d_sd) + cri * madv(d_sd)
        cutoff_od = median(d_od) + cri * madv(d_od)
    elseif cut == :q
        cutoff_sd = quantile(d_sd, 1 - risk)
        cutoff_od = quantile(d_od, 1 - risk)
    end
    d_sd ./= cutoff_sd
    d_od ./= cutoff_od
    d = [sqrt(d_sd[i] * d_od[i]) for i in eachindex(sd.dstand)]
    (d = d,)
end



