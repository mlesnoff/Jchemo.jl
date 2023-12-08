function gridcvbr(X, Y; segm, fun, score, pars, verbose = false)
    q = nco(Y)
    nrep = length(segm)
    res_rep = list(nrep)
    ncomb = length(pars[1]) # nb. combinations in pars
    @inbounds for i in 1:nrep
        verbose ? print("/ repl=", i, " ") : nothing
        listsegm = segm[i]       # segments in the repetition
        nsegm = length(listsegm) # segmts: =1; segmkf: =K
        zres = list(nsegm)       # results for the repetition
        @inbounds for j = 1:nsegm
            verbose ? print("segm=", j, " ") : nothing
            s = listsegm[j]
            zres[j] = gridscore(rmrow(X, s), rmrow(Y, s),
                X[s, :], Y[s, :];
                score = score, fun = fun, pars = pars)
        end
        zres = reduce(vcat, zres)
        dat = DataFrame(repl = fill(i, nsegm * ncomb),
            segm = repeat(1:nsegm, inner = ncomb))
        zres = hcat(dat, zres)
        res_rep[i] = zres
    end
    verbose ? println("/ End.") : nothing
    res_rep = reduce(vcat, res_rep)
    gdf = groupby(res_rep, collect(keys(pars))) 
    namy = map(string, repeat(["y"], q), 1:q)
    res = combine(gdf, namy .=> mean, renamecols = false)
    (res = res, res_rep = res_rep, )
end
    
