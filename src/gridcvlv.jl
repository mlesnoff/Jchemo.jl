"""
    gridcvlv(X, Y; segm, score, fun, nlv, pars, verbose = false)
* See `gridcv`.
* `nlv` : Nb., or collection of nb., of latent variables (LVs).

Same as [`gridcv`](@ref) but specific to (and much faster for) models 
using latent variables (e.g. PLSR).

Argument `pars` must not contain `nlv`.

See `?gridcv` for examples.
"""
function gridcvlv(X, Y; segm, score, fun, nlv, 
        pars = nothing, verbose = false)
    p = nco(X)
    q = nco(Y)
    nrep = length(segm)
    res_rep = list(nrep)
    nlv = max(0, minimum(nlv)):min(p, maximum(nlv))
    le_nlv = length(nlv)
    @inbounds for i in 1:nrep
        verbose ? print("/ repl=", i, " ") : nothing
        listsegm = segm[i]       # segments in the repetition
        nsegm = length(listsegm) # segmts: 1; segmkf: K
        zres = list(nsegm)       # results for the repetition
        @inbounds for j = 1:nsegm
            verbose ? print("segm=", j, " ") : nothing
            s = listsegm[j]
            zres[j] = gridscorelv(
                rmrow(X, s), rmrow(Y, s),
                X[s, :], Y[s, :];
                score = score, fun = fun, nlv = nlv, pars = pars)
        end
        zres = reduce(vcat, zres)
        ## Case where pars is empty
        if isnothing(pars) 
            dat = DataFrame(repl = fill(i, nsegm * le_nlv),
                segm = repeat(1:nsegm, inner = le_nlv))
        else
            ncomb = length(pars[1]) # nb. combinations in pars
            dat = DataFrame(repl = fill(i, nsegm * le_nlv * ncomb),
                segm = repeat(1:nsegm, inner = le_nlv * ncomb))
        end
        zres = hcat(dat, zres)
        res_rep[i] = zres
    end
    verbose ? println("/ End.") : nothing
    res_rep = reduce(vcat, res_rep)
    isnothing(pars) ? namgroup = [:nlv] : namgroup =  [:nlv ; collect(keys(pars))]
    gdf = groupby(res_rep, namgroup) 
    namy = map(string, repeat(["y"], q), 1:q)
    res = combine(gdf, namy .=> mean, renamecols = false)
    (res = res, res_rep = res_rep, )
end


