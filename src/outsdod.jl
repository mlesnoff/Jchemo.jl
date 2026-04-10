"""
    outsdod(fitm, X; typcut = :mad, cri = 3, alpha = .025)
Compute outlierness from PCA/PLS score and orthogonal distances (SD and OD).
* `fitm` : The preliminary model (e.g., object `fitm` returned by function `pcasvd`) that was fitted on 
    the data.
* `X` : X-data (n, p) on which was fitted the model `fitm`.
Keyword arguments:
* `typcut` : Type of cutoff to standardize SD and OD. Possible values are: `:mad`, `:q`. See Thereafter.
* `cri` : When `typcut` = `:mad`, a constant. See thereafter.
* `alpha` : When `typcut` = `:q`, a risk-I level. See thereafter.

In this method, outlierness `d` of a given observation is a consensus between the standardized score and
orthogonal distances. The returned consensus is computed by: 
* `d` = sqrt(SD_stand * OD_stand)
where:
* SD_stand = SD / cutoff_SD
* OD_stand = OD / cutoff_OD

The cutoff is computed with non-parametric heuristics. Noting [d] the SD- or OD-vector:
* If `typcut` = `:mad`, then cutoff = MED([d]) + `cri` * MAD([d]). 
* If `typcut` = `:q`, then cutoff is estimated from the empirical cumulative density function computed on [d], 
    for a given risk-I (`alpha`).

See function `outod` for examples.

## References
M. Hubert, V. J. Rousseeuw, K. Vanden Branden (2005). ROBPCA: a new approach to robust principal components analysis. 
Technometrics, 47, 64-79.

K. Vanden Branden, M. Hubert (2005). Robust classification in high dimension based on the SIMCA method. 
Chem. Lab. Int. Syst, 79, 10-21.

K. Varmuza, V. Filzmoser (2009). Introduction to multivariate statistical analysis in chemometrics. 
CRC Press, Boca Raton.
""" 
function outsdod(fitm, X; typcut = :mad, cri = 3, alpha = .025)
    @assert in(typcut, [:mad, :q]) "Argument 'typcut' must be :mad or :q."
    @assert 0 <= alpha <= 1 "Argument 'alpha' must ∈ [0, 1]."
    d_sd = outsd(fitm).d
    d_od = outod(fitm, X).d
    ## Since SD and OD have not the same scale, a scaling is required 
    ## to compute the SD-OD outlierness 'd' 
    if typcut == :mad
        cutoff_sd = median(d_sd) + cri * madv(d_sd)
        cutoff_od = median(d_od) + cri * madv(d_od)
    elseif typcut == :q
        cutoff_sd = quantile(d_sd, 1 - alpha)
        cutoff_od = quantile(d_od, 1 - alpha)
    end
    d_sd ./= cutoff_sd
    d_od ./= cutoff_od
    d = [sqrt(d_sd[i] * d_od[i]) for i in eachindex(d_sd)]
    (d = d,)
end



