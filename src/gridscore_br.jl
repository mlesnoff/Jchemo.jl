"""
    gridscore_br(Xtrain, Ytrain, X, Y; algo, score, pars, verbose = false)
Working function for `gridscore`.

See function `gridscore` for examples.
"""
function gridscore_br(Xtrain, Ytrain, X, Y; algo, score, pars, verbose = false)
    q = nco(Ytrain)
    ncomb = length(pars[1]) # nb. combinations in pars
    if verbose ; println("-- Nb. combinations = ", ncomb) ; end
    res = map(values(pars)...) do v...
        if verbose ; println(Pair.(keys(pars), v)...) ; end
        fitm = algo(Xtrain, Ytrain; Pair.(keys(pars), v)...)
        pred = predict(fitm, X).pred
        score(pred, Y)
    end
    if verbose ; println("-- End.") ; end
    res = ncomb == 1 ? res[1] : reduce(vcat, res) 
    dat = DataFrame(pars)
    namy = map(string, repeat(["y"], q), 1:q)
    res = DataFrame(res, Symbol.(namy))
    hcat(dat, res)
end


