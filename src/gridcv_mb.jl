"""
    gridcv_mb(Xbl, Y; segm, score, fun, pars, verbose = false)
* See `gridcv`.

Same as [`gridcv`](@ref) but specific to multiblock regression.

See `?gridcv` for examples.
"""
function gridcv_mb(Xbl, Y; segm, score, fun, pars, verbose = false)
    q = nco(Y)
    nrep = length(segm)
    res_rep = list(nrep)
    ncomb = length(pars[1]) # nb. combinations in pars
    nbl = length(Xbl)
    @inbounds for i in 1:nrep
        verbose ? print("/ rept=", i, " ") : nothing
        listsegm = segm[i]       # segments in the repetition
        nsegm = length(listsegm) # segmts: 1; segmkf: K
        zres = list(nsegm)       # results for the repetition
        @inbounds for j = 1:nsegm
            verbose ? print("segm=", j, " ") : nothing
            s = listsegm[j]
            zX1 = list(nbl, Matrix{Float64})
            zX2 = list(nbl, Matrix{Float64})
            for k = 1:nbl
                zX1[k] = rmrow(Xbl[k], s)
                zX2[k] = Xbl[k][s, :]
            end
            zres[j] = gridscore(zX1, rmrow(Y, s), zX2, Y[s, :];
                score = score, fun = fun, pars = pars)
        end
        zres = reduce(vcat, zres)
        dat = DataFrame(rept = fill(i, nsegm * ncomb),
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

"""
    gridcvlv_mb(Xbl, Y; segm, score, fun, nlv, pars, verbose = false)
* See `gridcv`.

Same as [`gridcvlv`](@ref) but specific to multiblock regression.

See `?gridcv` for examples.
"""
function gridcvlv_mb(Xbl, Y; segm, score, fun, nlv, 
        pars = nothing, verbose = false)
    q = nco(Y)
    nrep = length(segm)
    res_rep = list(nrep)
    nlv = max(minimum(nlv), 0):maximum(nlv)
    le_nlv = length(nlv)
    nbl = length(Xbl)
    @inbounds for i in 1:nrep
        verbose ? print("/ rept=", i, " ") : nothing
        listsegm = segm[i]       # segments in the repetition
        nsegm = length(listsegm) # segmts: 1; segmkf: K
        zres = list(nsegm)       # results for the repetition
        @inbounds for j = 1:nsegm
            verbose ? print("segm=", j, " ") : nothing
            s = listsegm[j]
            zX1 = list(nbl, Matrix{Float64})
            zX2 = list(nbl, Matrix{Float64})
            for k = 1:nbl
                zX1[k] = rmrow(Xbl[k], s)
                zX2[k] = Xbl[k][s, :]
            end
            zres[j] = gridscorelv(zX1, rmrow(Y, s), zX2, Y[s, :];
                score = score, fun = fun, nlv = nlv, pars = pars)
        end
        zres = reduce(vcat, zres)
        ## Case where pars is empty
        if isnothing(pars) 
            dat = DataFrame(rept = fill(i, nsegm * le_nlv),
                segm = repeat(1:nsegm, inner = le_nlv))
        else
            ncomb = length(pars[1]) # nb. combinations in pars
            dat = DataFrame(rept = fill(i, nsegm * le_nlv * ncomb),
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


