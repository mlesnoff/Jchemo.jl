"""
    occsdod(; kwargs...)
    occsdod(object, X; kwargs...)
One-class classification (OCC) using a consensus between PCA/PLS score and orthogonal distances (SD and OD).
* `fitm` : The preliminary model (e.g., object `fitm` returned by function `pcasvd`) that was fitted on 
    the training data assumed to represent the reference (= target) class.
* `X` : Training X-data (n, p) on which was fitted model `fitm`.
Keyword arguments:
* `typcut` : Type of cutoff. Possible values are: `:mad`, `:q`. See Thereafter.
* `cri` : When `typcut` = `:mad`, a constant. See thereafter.
* `alpha` : When `typcut` = `:q`, a risk-I level. See thereafter.

In this function, outlierness `d` of a given observation is a consensus between the score distance (SD) and the
orthogonal distance (OD). The consensus is computed from the standardized distances by: 
* `dstand` = sqrt(`dstand_sd` * `dstand_od`).

See functions:
* `occsd` for details on the cutoff computation and the outputs,
* and `occod` for examples.
""" 
occsdod(; kwargs...) = JchemoModel(occsdod, nothing, kwargs)

function occsdod(fitm, X; kwargs...) 
    par = recovkw(ParOccsdod, kwargs).par 
    @assert 0 <= par.gamma <= 1 "Argument 'gamma' must ∈ [0, 1]."    
    #fitmsd = occsd(fitm; typcut = par.typcut, cri = par.cri, ampha = par.alpha)
    #fitmod = occod(fitm, X; typcut = par.typcut, cri = par.cri, ampha = par.alpha)
    #sd = fitmsd.d
    #od = fitmod.d
    #z = [sqrt(sd.dstand[i] * od.dstand[i]) for i in eachindex(sd.d)]
    #nam = string.(names(sd), "_sd")
    #rename!(sd, nam)
    #nam = string.(names(od), "_od")
    #rename!(od, nam)
    #d = hcat(sd, od)
    #d.dstand = z
    #Occsdod(d, fitmsd, fitmod, par)
    sd = outsd(fitm)
    od = outod(fitm, X)
    sdod = outsdod(fitm, X; gamma = par.gamma, fscal = par.fscal)
    ##
    d = sdod.d
    if par.typcut == :mad
        cutoff = median(d) + par.cri * madv(d)
    elseif par.typcut == :q
        cutoff = quantile(d, 1 - par.alpha)
    end
    e_cdf = StatsBase.ecdf(d)
    d = DataFrame(
        d = d, 
        dstand = d / cutoff, 
        pval = pval(e_cdf, d), 
        )
    Occsdod(d, fitm, e_cdf, cutoff, sd, od, sdod, par)
end

"""
    predict(object::Occsdod, X)
Compute predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Occsdod, X)
    tscales = object.sd.tscales    
    gamma = object.par.gamma 
    sigma_sd = object.sdod.sigma_sd
    sigma_od = object.sdod.sigma_od
    ## SD
    T = transf(object.fitm, X)
    Q = eltype(T)
    m, nlv = size(T)
    fscale!(T, tscales)
    d2 = vec(eucl2(T, zeros(Q, nlv)'))
    sd = sqrt.(d2)
    ## OD
    E = xresid(object.fitm, X)
    od = rownorm(E)
    ## Consensus
    d = gamma * sd / sigma_sd + (1 - gamma) * od / sigma_od
    ## End
    d = DataFrame(
        d = d, 
        dstand = d / object.cutoff, 
        pval = pval(object.e_cdf, d), 
        gh = d2 / nlv
        )
    pred = [if d.dstand[i] <= 1 "in" else "out" end for i in eachindex(d.d)]
    pred = reshape(pred, m, 1)
    (pred = pred, d)
end

