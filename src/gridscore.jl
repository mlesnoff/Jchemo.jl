"""
    gridscore(Xtrain, Ytrain, X, Y ; score, fun, pars, verbose = FALSE) 
Compute a prediction score (error rate; e.g. RMSEP) for a given model over a grid of parameter values.
- Xtrain : Training X-data (n, p) or (n,).
- Ytrain : Training Y-data (n, q) or (n,).
- X : Validation X-data (m, p) or (m,).
- Y : Validation Y-data (m, q) or (m,).
- score : Function computing the score.
- fun : Function computing the model.
- pars : tuple of named vectors (arguments of fun) of same length,
involved in the calculation of the score.
- verbose : If true, fitting information are printed.

The scores are computed on X and Y for each combination of the grid defined in pars. 
    
The output is a dataframe.
"""
function gridscore(Xtrain, Ytrain, X, Y ; score, fun, pars, verbose = false)
    m = size(X, 1)
    q = size(Ytrain, 2)
    nco = length(pars[1])
    verbose ? println("-- Nb. combinations = ", nco) : nothing
    res = map(values(pars)...) do v...
        verbose ? println(Pair.(keys(pars), v)...) : nothing
        fm = fun(Xtrain, Ytrain ; Pair.(keys(pars), v)...)
        pred = predict(fm, X).pred
        score(pred, Y)
    end
    verbose ? println("-- End.") : nothing
    ## Case nb. comb > 1 ==> vertical concatenation
    #if isa(res, Vector)
    if nco > 1
        res = reduce(vcat, res) 
    end
    ## End
    namy = map(string, repeat(["y"], q), 1:q)
    res = DataFrame(res, Symbol.(namy))
    dat = DataFrame(pars)
    res = hcat(dat, res)
    res
end

"""
    gridscorelv(Xtrain, Ytrain, X, Y ; score, fun, nlv, pars, verbose = FALSE)
Same as [`gridscore`](@ref) but specific (and much faster) to LV-based models (e.g. PLSR).
- nlv : Nb. or collection of nb. of latent variables (LVs).

Argument pars must not contain nlv.
"""
function gridscorelv(Xtrain, Ytrain, X, Y ; score, fun, nlv, pars = nothing, verbose = false)
    m = size(X, 1)
    q = size(Ytrain, 2)
    nlv = max(minimum(nlv), 0):maximum(nlv)
    le_nlv = length(nlv)
    namy = map(string, repeat(["y"], q), 1:q)
    if isnothing(pars)
        verbose ? println("-- Nb. combinations = 0.") : nothing
        fm = fun(Xtrain, Ytrain, nlv = maximum(nlv))
        pred = predict(fm, X, nlv = nlv).pred
        le_nlv == 1 ? pred = [pred] : nothing
        res = zeros(le_nlv, q)
        for i = 1:le_nlv
            res[i, :] = score(pred[i], Y)
        end
        dat = DataFrame(nlv = nlv)
    else
        nco = length(pars[1])
        verbose ? println("-- Nb. combinations = ", nco) : nothing
        res = map(values(pars)...) do v...
            fm = fun(Xtrain, Ytrain ; nlv = maximum(nlv), Pair.(keys(pars), v)...)
            pred = predict(fm, X ; nlv).pred
            le_nlv == 1 ? pred = [pred] : nothing
            zres = zeros(le_nlv, q)
            for i = 1:le_nlv
                zres[i, :] = score(pred[i], Y)
            end
            zres
        end 
        ## Case nb. comb > 1 ==> vertical concatenation
        if isa(res, Vector)
            res = reduce(vcat, res)
        end
        ## End
        ## ==> To build for cases where pars is not nothing: dat
    end
    verbose ? println("-- End.") : nothing
    res = DataFrame(res, Symbol.(namy))
    res = hcat(dat, res)
    res
end

