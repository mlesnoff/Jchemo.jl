"""
    gridcv(X, Y; segm, score, fun, pars, verbose = false) 
Cross-validation (CV) over a grid of parameters.
* `X` : X-data.
* `Y` : Y-data.
* `segm` : Segments of the CV (output of functions
     [`segmts`](@ref), [`segmkf`](@ref) etc.).
* `score` : Function computing the prediction score (= error rate; e.g. MSEP).
* `fun` : Function computing the prediction model.
* `pars` : tuple of named vectors (arguments of `fun`) 
    defining the grid of parameters.
* `verbose` : If true, fitting information are printed.

The score is computed over X and Y for each combination of the 
grid defined in `pars`. 

The vectors in `pars` must have same length.
"""
function gridcv(X, Y; segm, score, fun, pars, verbose = false)
    q = size(Y, 2)
    nrep = length(segm)
    res_rep = list(nrep)
    nco = length(pars[1]) # nb. combinations in pars
    @inbounds for i in 1:nrep
        verbose ? print("/ rept=", i, " ") : nothing
        listsegm = segm[i]       # segments in the repetition
        nsegm = length(listsegm) # segmts: 1; segmkf: K
        zres = list(nsegm)       # results for the repetition
        @inbounds for j = 1:nsegm
            verbose ? print("segm=", j, " ") : nothing
            s = listsegm[j]
            zres[j] = gridscore(
                rmrow(X, s), rmrow(Y, s),
                X[s, :], Y[s, :];
                score = score, fun = fun, pars = pars)
        end
        zres = reduce(vcat, zres)
        dat = DataFrame(rept = fill(i, nsegm * nco),
            segm = repeat(1:nsegm, inner = nco))
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
    gridcvlv(X, Y; segm, score, fun, nlv, pars, verbose = false)
* `nlv` : Nb., or collection of nb., of latent variables (LVs).

Same as [`gridcv`](@ref) but specific to (and much faster for) models 
using latent variables (e.g. PLSR).

Argument `pars` must not contain `nlv`.
"""
function gridcvlv(X, Y; segm, score, fun, nlv, pars = nothing, 
        verbose = false)
    q = size(Y, 2)
    nrep = length(segm)
    res_rep = list(nrep)
    nlv = max(minimum(nlv), 0):maximum(nlv)
    le_nlv = length(nlv)
    @inbounds for i in 1:nrep
        verbose ? print("/ rept=", i, " ") : nothing
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
            dat = DataFrame(rept = fill(i, nsegm * le_nlv),
                segm = repeat(1:nsegm, inner = le_nlv))
        else
            nco = length(pars[1]) # nb. combinations in pars
            dat = DataFrame(rept = fill(i, nsegm * le_nlv * nco),
                segm = repeat(1:nsegm, inner = le_nlv * nco))
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

"""
    gridcvlb(X, Y; segm, score, fun, lb, pars, verbose = false)
* `nlv` : Nb., or collection of nb., of latent variables (LVs).

Same as [`gridcv`](@ref) but specific to (and much faster for) models 
using ridge regularization (e.g. RR).

Argument `pars` must not contain `lb`.
"""
function gridcvlb(X, Y; segm, score, fun, lb, pars = nothing, 
        verbose = false)
    q = size(Y, 2)
    nrep = length(segm)
    res_rep = list(nrep)
    lb = sort(unique(lb))
    le_lb = length(lb)
    @inbounds for i in 1:nrep
        verbose ? print("/ rept=", i, " ") : nothing
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
            dat = DataFrame(rept = fill(i, nsegm * le_lb),
                segm = repeat(1:nsegm, inner = le_lb))
        else
            nco = length(pars[1]) # nb. combinations in pars
            dat = DataFrame(rept = fill(i, nsegm * le_lb * nco),
                segm = repeat(1:nsegm, inner = le_lb * nco))
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

####################### Multiblock

function gridcv_mb(X, Y; segm, score, fun, pars, verbose = false)
    q = size(Y, 2)
    nrep = length(segm)
    res_rep = list(nrep)
    nco = length(pars[1]) # nb. combinations in pars
    nbl = length(X)
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
                zX1[k] = rmrow(X[k], s)
                zX2[k] = X[k][s, :]
            end
            zres[j] = gridscore(zX1, rmrow(Y, s), zX2, Y[s, :];
                score = score, fun = fun, pars = pars)
        end
        zres = reduce(vcat, zres)
        dat = DataFrame(rept = fill(i, nsegm * nco),
            segm = repeat(1:nsegm, inner = nco))
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

function gridcvlv_mb(X, Y; segm, score, fun, nlv, pars = nothing, 
        verbose = false)
    q = size(Y, 2)
    nrep = length(segm)
    res_rep = list(nrep)
    nlv = max(minimum(nlv), 0):maximum(nlv)
    le_nlv = length(nlv)
    nbl = length(X)
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
                zX1[k] = rmrow(X[k], s)
                zX2[k] = X[k][s, :]
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
            nco = length(pars[1]) # nb. combinations in pars
            dat = DataFrame(rept = fill(i, nsegm * le_nlv * nco),
                segm = repeat(1:nsegm, inner = le_nlv * nco))
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




