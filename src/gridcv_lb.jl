"""
    gridcv_lb(X, Y; segm, algo, score, pars = nothing, lb, verbose = false)
Working function for `gridcv`.

Specific and faster than `gridcv_br` for models using ridge regularization (e.g. RR). Argument `pars` must 
not contain `nlv`.

See function `gridcv` for examples.
"""
function gridcv_lb(X, Y; segm, algo, score, pars = nothing, lb, verbose = false)
    q = nco(Y)
    nrep = length(segm)
    res_rep = list(nrep)
    lb = mlev(lb)
    le_lb = length(lb)
    @inbounds for i in 1:nrep
        verbose ? print("/ rep=", i, " ") : nothing
        listsegm = segm[i]       # segments in the repetition
        nsegm = length(listsegm) # segmts: = 1; segmkf: = K
        zres = list(nsegm)       # results for the repetition
        @inbounds for j in eachindex(listsegm) # = 1:nsegm
            verbose ? print("segm=", j, " ") : nothing
            s = listsegm[j]

            if isa(X[1, 1], Number)  # monoblock
                zres[j] = gridscore_lb(rmrow(X, s), rmrow(Y, s), X[s, :], Y[s, :]; algo, score, pars, lb)
            else                     # multiblock
                Xcal = similar(X)
                Xval = similar(X)
                @inbounds for k in eachindex(X) 
                    Xcal[k] = rmrow(X[k], s)
                    Xval[k] = X[k][s, :]
                end
                zres[j] = gridscore_lb(Xcal, rmrow(Y, s), Xval, Y[s, :]; algo, score, pars, lb)
            end
        end
        zres = reduce(vcat, zres)
        ## Case where pars is empty
        if isnothing(pars) 
            dat = DataFrame(rep = fill(i, nsegm * le_lb), segm = repeat(1:nsegm, inner = le_lb))
        else
            ncomb = length(pars[1]) # nb. combinations in pars
            dat = DataFrame(rep = fill(i, nsegm * le_lb * ncomb), segm = repeat(1:nsegm, inner = le_lb * ncomb))
        end
        zres = hcat(dat, zres)
        res_rep[i] = zres
    end
    verbose ? println("/ End.") : nothing
    res_rep = reduce(vcat, res_rep)
    isnothing(pars) ? namgroup = [:lb] : namgroup = [:lb ; collect(keys(pars))]
    gdf = groupby(res_rep, namgroup) 
    namy = map(string, repeat(["y"], q), 1:q)
    res = combine(gdf, namy .=> mean, renamecols = false)
    (res = res, res_rep)
end


