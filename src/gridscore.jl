"""
    gridscore(Xtrain, Ytrain, X, Y; score, fun, pars, verbose = FALSE) 
Model validation over a grid of parameters.
* `Xtrain` : Training X-data.
* `Ytrain` : Training Y-data.
* `X` : Validation X-data.
* `Y` : Validation Y-data.
* `score` : Function computing the prediction score (= error rate; e.g. msep).
* `fun` : Function computing the prediction model.
* `pars` : tuple of named vectors (arguments of fun) of same length
    involved in the calculation of the score.
* `verbose` : If true, fitting information are printed.

Compute a prediction score for a given model over a grid of parameters.

The score is computed over X and Y for each combination of the 
grid defined in `pars`. 
    
The vectors in `pars` must have same length.
"""
function gridscore(Xtrain, Ytrain, X, Y; score, fun, 
        pars, verbose = false)
    q = nco(Ytrain)
    ncomb = length(pars[1]) # nb. combinations in pars
    verbose ? println("-- Nb. combinations = ", ncomb) : nothing
    res = map(values(pars)...) do v...
        verbose ? println(Pair.(keys(pars), v)...) : nothing
        fm = fun(Xtrain, Ytrain ; Pair.(keys(pars), v)...)
        pred = predict(fm, X).pred
        score(pred, Y)
    end
    verbose ? println("-- End.") : nothing
    ncomb == 1 ? res = res[1] : res = reduce(vcat, res) 
    dat = DataFrame(pars)
    namy = map(string, repeat(["y"], q), 1:q)
    res = DataFrame(res, Symbol.(namy))
    hcat(dat, res)
end

"""
    gridscorelv(Xtrain, Ytrain, X, Y; score, fun, nlv, pars, verbose = FALSE)
* `nlv` : Nb., or collection of nb., of latent variables (LVs).

Same as [`gridscore`](@ref) but specific to (and much faster for) models 
using latent variables (e.g. PLSR).

Argument `pars` must not contain `nlv`.
"""
function gridscorelv(Xtrain, Ytrain, X, Y; score, fun, nlv, 
        pars = nothing, verbose = false)
    p = nco(Xtrain)
    q = nco(Ytrain)
    nlv = max(0, minimum(nlv)):min(p, maximum(nlv))
    le_nlv = length(nlv)
    if isnothing(pars)
        verbose ? println("-- Nb. combinations = 0.") : nothing
        fm = fun(Xtrain, Ytrain, nlv = maximum(nlv))
        pred = predict(fm, X; nlv = nlv).pred
        le_nlv == 1 ? pred = [pred] : nothing
        res = zeros(le_nlv, q)
        @inbounds for i = 1:le_nlv
            res[i, :] = score(pred[i], Y)
        end
        dat = DataFrame(nlv = nlv)
    else       
        ncomb = length(pars[1])  # nb. combinations in pars
        verbose ? println("-- Nb. combinations = ", ncomb) : nothing
        res = map(values(pars)...) do v...    
            verbose ? println(Pair.(keys(pars), v)...) : nothing
            fm = fun(Xtrain, Ytrain ; nlv = maximum(nlv), Pair.(keys(pars), v)...)
            pred = predict(fm, X; nlv = nlv).pred
            le_nlv == 1 ? pred = [pred] : nothing
            zres = zeros(le_nlv, q)
            for i = 1:le_nlv
                zres[i, :] = score(pred[i], Y)
            end
            zres
        end 
       
        ncomb == 1 ? res = res[1] : res = reduce(vcat, res) 
        ## Make dat
        if le_nlv == 1
            dat = DataFrame(pars)
        else
            zdat = DataFrame(pars)
            dat = list(ncomb)
            @inbounds for i = 1:ncomb
                dat[i] = reduce(vcat, fill(zdat[i:i, :], le_nlv))
            end
            dat = reduce(vcat, dat)
        end
        znlv = repeat(nlv, ncomb)
        dat = hcat(dat, DataFrame(nlv = znlv))
        ## End
    end
    verbose ? println("-- End.") : nothing
    namy = map(string, repeat(["y"], q), 1:q)
    res = DataFrame(res, Symbol.(namy))
    hcat(dat, res)
end

"""
    gridscorelb(Xtrain, Ytrain, X, Y; score, fun, lb, pars, verbose = FALSE)
* `lb` : Value, or collection of values, of the regularization parameter "lambda".

Same as [`gridscore`](@ref) but specific to (and much faster for) models 
using ridge regularization (e.g. RR).

Argument `pars` must not contain `lb`.
"""
function gridscorelb(Xtrain, Ytrain, X, Y; score, fun, lb, 
        pars = nothing, verbose = false)
    q = nco(Ytrain)
    lb = sort(unique(lb))
    le_lb = length(lb)
    if isnothing(pars)
        verbose ? println("-- Nb. combinations = 0.") : nothing
        fm = fun(Xtrain, Ytrain, lb = maximum(lb))
        pred = predict(fm, X; lb = lb).pred
        le_lb == 1 ? pred = [pred] : nothing
        res = zeros(le_lb, q)
        @inbounds for i = 1:le_lb
            res[i, :] = score(pred[i], Y)
        end
        dat = DataFrame(lb = lb)
    else
        ncomb = length(pars[1])  # nb. combinations in pars
        verbose ? println("-- Nb. combinations = ", ncomb) : nothing
        res = map(values(pars)...) do v...
            verbose ? println(Pair.(keys(pars), v)...) : nothing
            fm = fun(Xtrain, Ytrain ; lb = maximum(lb), Pair.(keys(pars), v)...)
            pred = predict(fm, X; lb = lb).pred
            le_lb == 1 ? pred = [pred] : nothing
            zres = zeros(le_lb, q)
            for i = 1:le_lb
                zres[i, :] = score(pred[i], Y)
            end
            zres
        end 
        ncomb == 1 ? res = res[1] : res = reduce(vcat, res) 
        ## Make dat
        if le_lb == 1
            dat = DataFrame(pars)
        else
            zdat = DataFrame(pars)
            dat = list(ncomb)
            @inbounds for i = 1:ncomb
                dat[i] = reduce(vcat, fill(zdat[i:i, :], le_lb))
            end
            dat = reduce(vcat, dat)
        end
        zlb = repeat(lb, ncomb)
        dat = hcat(dat, DataFrame(lb = zlb))
        ## End
    end
    verbose ? println("-- End.") : nothing
    namy = map(string, repeat(["y"], q), 1:q)
    res = DataFrame(res, Symbol.(namy))
    hcat(dat, res)
end



