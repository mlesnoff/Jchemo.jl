"""
    outsdod(fitm, X; nlv::Int = nco(fitm.T), gamma::Q = .5, fscal::Function = madv) where Q <: Float
Compute outlierness from PCA/PLS score (SD) and orthogonal (OD) distances.
* `fitm` : The reduction dimension model that was fitted on the data (e.g., object `fitm` returned by functions 
    `pcasvd` or `plskern`).
* `X` : X-data (n, p) on which was fitted model `fitm`.
Keyword arguments:
* `nlv` : Nb. latent variables (LVs) to consider. By default, it is the maximum nb. of LVs
    defined in model `fitm`.
* `gamma` : Proportion (∈ [0, 1]) of scaled SD in the consensus (see below).
* `fscal` : Function used to scale SD and OD in the consensus.

This function computes outlierness `d` of each observation (row) of `X` by a *consensus* (weighted mean of scaled SD and OD
of the observation): 
* `d` = `gamma` * SD / `fscal`(SD) + (1 - `gamma`) * OD / `fscal`(OD) 
where `gamma` ∈ [0, 1]. The scaling (`fscal`) ensures that SD and OD have the same order of magnitude before 
the consensus. 

See functions `outsd` and `outod` for details on SD and OD, and function `outod` for examples.
""" 
function outsdod(fitm, X; nlv::Int = nco(fitm.T), gamma::Q = .5, fscal::Function = madv) where Q <: Float
    X = ensure_mat(X)
    gamma = Q(gamma)
    sd = outsd(fitm; nlv).d
    od = outod(fitm, X; nlv).d
    sigma_sd = fscal(sd)
    sigma_od = fscal(od) 
    d = gamma * sd / sigma_sd + (1 - gamma) * od / sigma_od
    (d = d, sigma_sd, sigma_od)
end

