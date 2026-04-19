"""
    gridcv(model::Pipeline, Xtrain, Ytrain; segm, score, pars = nothing, nlv = nothing, lb = nothing, 
        verbose = false) 
Cross-validation (CV) of a pipeline model over a grid of parameters.
* `model` : Model to evaluate.
* `X` : Training X-data (n, p).
* `Y` : Training Y-data (n, q).
Keyword arguments: 
* `segm` : Segments of observations used for the CV (output of functions [`segmts`](@ref), [`segmkf`](@ref), etc.).
* `score` : Function computing the prediction score (e.g., `rmsep`).
* `pars` : tuple of named vectors of same length defining the parameter combinations (e.g., output of function `mpar`).
    Must only be parameters for the last model of the pipeline (= the final predictor).
* `verbose` : If `true`, predicting information are printed.
* `nlv` : Value, or vector of values, of the nb. of latent variables (LVs).
* `lb` : Value, or vector of values, of the ridge regularization parameter "lambda".

**Note**: In the present version of the function, only the last model of the pipeline (= the final predictor) 
is tuned.

For other details, see function `gridcv` for non-pipeline models. 

## Examples
```julia

```
"""
function  gridcv(model, X, Y; segm, score, pars = nothing, nlv = nothing, lb = nothing,  #::Pipeline
        verbose = false) 
    q = nco(Y)
    nrep = length(segm)
    res_rep = list(nrep)
    @inbounds for i in 1:nrep
        verbose ? print("/ rep=", i, " ") : nothing
        listsegm = segm[i]       # segments in the repetition
        nsegm = length(listsegm) # segmts: = 1; segmkf: = K
        zres = list(nsegm)       # results for the repetition
        @inbounds for j = 1:nsegm
            verbose ? print("segm=", j, " ") : nothing
            s = listsegm[j]
            ## Monobloc
            if isa(X[1, 1], Number)
                zres[j] = gridscore(model, rmrow(X, s), rmrow(Y, s), X[s, :], Y[s, :]; score, pars, nlv, lb)
            ## Multiblock
            else  
                Xcal = similar(X)
                Xval = similar(X)
                @inbounds for k in eachindex(X) 
                    Xcal[k] = rmrow(X[k], s)
                    Xval[k] = X[k][s, :]
                end
                zres[j] = gridscore(model, Xcal, Ycal, Xval, Yval; score, pars, nlv, lb)
            end
        end
        ncomb = nro(zres[1])
        zres = reduce(vcat, zres)
        dat = DataFrame(rep = fill(i, nsegm * ncomb), segm = repeat(1:nsegm, inner = ncomb))
        zres = hcat(dat, zres)
        res_rep[i] = zres
    end
    verbose ? println("/ End.") : nothing
    res_rep = reduce(vcat, res_rep)
    ## Average scores over reps and segms
    if isnothing(nlv) && isnothing(lb)
        gdf = groupby(res_rep, collect(keys(pars))) 
    elseif !isnothing(nlv)
        isnothing(pars) ? namgroup = [:nlv] : namgroup = [:nlv ; collect(keys(pars))]
        gdf = groupby(res_rep, namgroup) 
    elseif !isnothing(lb)
        isnothing(pars) ? namgroup = [:lb] : namgroup = [:lb ; collect(keys(pars))]
        gdf = groupby(res_rep, namgroup) 
    end
    namy = map(string, repeat(["y"], q), 1:q)
    res = combine(gdf, namy .=> meanv, renamecols = false)
    ## End
    (res = res, res_rep)
end

