"""
    gridcv_lv((X, Y; segm, algo, score, pars = nothing, nlv, verbose = false)
Working function for `gridcv`.

Specific and faster than `gridcv_br` for models 
using latent variables (e.g. PLSR). Argument `pars` 
must not contain `nlv`.

See function `gridcv` for examples.
"""
function gridcv_lv(X, Y; segm, algo, score, pars = nothing, nlv, verbose = false)
    p = nco(X)
    q = nco(Y)
    nrep = length(segm)
    res_rep = list(nrep)
    nlv = max(0, minimum(nlv)):min(p, maximum(nlv))
    le_nlv = length(nlv)
    @inbounds for i in 1:nrep
        verbose ? print("/ rep=", i, " ") : nothing
        listsegm = segm[i]       # segments in the repetition
        nsegm = length(listsegm) # segmts: 1; segmkf: K
        zres = list(nsegm)       # results for the repetition
        @inbounds for j = 1:nsegm
            verbose ? print("segm=", j, " ") : nothing
            s = listsegm[j]
            zres[j] = gridscore_lv(rmrow(X, s), rmrow(Y, s), X[s, :], Y[s, :]; algo, score, nlv, pars)
        end
        zres = reduce(vcat, zres)
        ## Case where pars is empty
        if isnothing(pars) 
            dat = DataFrame(rep = fill(i, nsegm * le_nlv), segm = repeat(1:nsegm, inner = le_nlv))
        else
            ncomb = length(pars[1]) # nb. combinations in pars
            dat = DataFrame(rep = fill(i, nsegm * le_nlv * ncomb), segm = repeat(1:nsegm, inner = le_nlv * ncomb))
        end
        zres = hcat(dat, zres)
        res_rep[i] = zres
    end
    verbose ? println("/ End.") : nothing
    res_rep = reduce(vcat, res_rep)
    isnothing(pars) ? namgroup = [:nlv] : namgroup = [:nlv ; collect(keys(pars))]
    gdf = groupby(res_rep, namgroup) 
    namy = map(string, repeat(["y"], q), 1:q)
    res = combine(gdf, namy .=> mean, renamecols = false)
    (res = res, res_rep)
end


