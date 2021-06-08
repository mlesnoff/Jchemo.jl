"""
    gridscore(Xtrain, Ytrain, X, Y ; score, fun, pars, verbose = FALSE)
pars: tuple of named vectors (arguments of fun) of same length,
involved in the calculation of the score.
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
pars: tuple of named vectors (arguments of fun) of same length,
involved in the calculation of the score. The tuple must not contain nlv.
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

