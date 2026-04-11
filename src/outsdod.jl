"""
    outsdod(fitm, X; gamma = .5, fscal = madv)
Compute outlierness from PCA/PLS score and orthogonal distances (SD and OD).
* `fitm` : The reduction dimension model that was fitted on the data (e.g., object `fitm` returned by functions 
    `pcasvd` or `plskern`).
* `X` : X-data (n, p) on which was fitted model `fitm`.
Keyword arguments:
* `gamma` : Proportion of scaled SD in the consensus (see below).
* `fscal` : Function used to scale SD and OD (by defalut, this is `madv`). 

In this method, outlierness `d` of a given observation is defined as a consensus between scaled SD and OD. 
The returned consensus is computed by: 
* `d` = `gamma` * SD / `fscal`(SD) + (1 - `gamma`) * OD / `fscal`(OD) 

See function `outod` for examples.
""" 
function outsdod(fitm, X; gamma = .5, fscal = madv)
    res = outsd(fitm)
    sd = res.d
    tscales = res.tscales
    od = outod(fitm, X).d
    sigma_sd = fscal(sd)
    sigma_od = fscal(od)
    #d = sqrt.(gamma * sd.^2 / fscal(sd.^2) + (1 - gamma) * od.^2 / fscal(od.^2))   
    d = gamma * sd / sigma_sd + (1 - gamma) * od / sigma_od
    (d = d, tscales, sigma_sd, sigma_od)
end

