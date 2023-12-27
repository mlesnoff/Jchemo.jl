"""
    gridcv_br(X, Y; segm, fun, 
        score, pars, verbose = false)
Working function for `gridcv`.

See function `gridcv` for examples.
"""
function gridcv_br(X, Y; segm, fun, 
        score, pars, verbose = false)
    q = nco(Y)
    nrep = length(segm)
    res_rep = list(nrep)
    ncomb = length(pars[1])      # nb. combinations in pars
    @inbounds for i in 1:nrep
        verbose ? print("/ rep=", i, " ") : nothing
        listsegm = segm[i]       # segments in the repetition
        nsegm = length(listsegm) # segmts: =1; segmkf: =K
        zres = list(nsegm)       # results for the repetition
        @inbounds for j = 1:nsegm
            verbose ? print("segm=", j, " ") : nothing
            s = listsegm[j]
            zres[j] = gridscore_br(rmrow(X, s), rmrow(Y, s),
                X[s, :], Y[s, :]; fun, score, pars)
        end
        zres = reduce(vcat, zres)
        dat = DataFrame(rep = fill(i, nsegm * ncomb),
            segm = repeat(1:nsegm, inner = ncomb))
        zres = hcat(dat, zres)
        res_rep[i] = zres
    end
    verbose ? println("/ End.") : nothing
    res_rep = reduce(vcat, res_rep)
    gdf = groupby(res_rep, collect(keys(pars))) 
    namy = map(string, repeat(["y"], q), 1:q)
    res = combine(gdf, namy .=> mean, 
        renamecols = false)
    (res = res, res_rep)
end
    
