"""
    gridcvlb(X, Y; segm, score, fun, lb, pars, verbose = false)
* See `gridcv`.
* `lb` : Value, or collection of values, of the ridge regularization parameter "lambda".

Same as [`gridcv`](@ref) but specific to (and much faster for) models 
using ridge regularization (e.g. RR).

Argument `pars` must not contain `lb`.

See `?gridcv` for examples.
"""
function gridcvlb(X, Y; segm, score, fun, lb, 
        pars = nothing, verbose = false)
    q = nco(Y)
    nrep = length(segm)
    res_rep = list(nrep)
    lb = mlev(lb)
    le_lb = length(lb)
    @inbounds for i in 1:nrep
        verbose ? print("/ repl=", i, " ") : nothing
        listsegm = segm[i]       # segments in the repetition
        nsegm = length(listsegm) # segmts: 1; segmkf: K
        zres = list(nsegm)       # results for the repetition
        @inbounds for j = 1:nsegm
            verbose ? print("segm=", j, " ") : nothing
            s = listsegm[j]
            zres[j] = gridscorelb(
                rmrow(X, s), rmrow(Y, s),
                X[s, :], Y[s, :];
                score = score, fun = fun, lb = lb, pars = pars)
        end
        zres = reduce(vcat, zres)
        ## Case where pars is empty
        if isnothing(pars) 
            dat = DataFrame(repl = fill(i, nsegm * le_lb),
                segm = repeat(1:nsegm, inner = le_lb))
        else
            ncomb = length(pars[1]) # nb. combinations in pars
            dat = DataFrame(repl = fill(i, nsegm * le_lb * ncomb),
                segm = repeat(1:nsegm, inner = le_lb * ncomb))
        end
        zres = hcat(dat, zres)
        res_rep[i] = zres
    end
    verbose ? println("/ End.") : nothing
    res_rep = reduce(vcat, res_rep)
    isnothing(pars) ? namgroup = [:lb] : namgroup =  [:lb ; collect(keys(pars))]
    gdf = groupby(res_rep, namgroup) 
    namy = map(string, repeat(["y"], q), 1:q)
    res = combine(gdf, namy .=> mean, renamecols = false)
    (res = res, res_rep = res_rep, )
end


