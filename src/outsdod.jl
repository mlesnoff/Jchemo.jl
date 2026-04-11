"""
    outsdod(fitm, X; gamma = .5, fscal = madv)
Compute outlierness from PCA/PLS score and orthogonal distances (SD and OD).
* `fitm` : The reduction dimension model that was fitted on the data (e.g., object `fitm` returned by functions 
    `pcasvd` or `plskern`).
* `X` : X-data (n, p) on which was fitted model `fitm`.

In this method, outlierness `d` of a given observation is a consensus between scaled SD and OD. The returned consensus 
is computed by: 
* `d` = `gamma` * SD / `fscal`(SD) + (1 - `gamma`) * OD / `fscal`(OD) 

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



