"""
    gridscore_br(Xtrain, Ytrain, X, Y; fun, score, pars, 
        verbose = false)
Working function for `gridscore`.

See function `gridscore` for examples.
"""
function gridscore_br(Xtrain, Ytrain, X, Y; fun, score, pars, 
        verbose = false)
    q = nco(Ytrain)
    ncomb = length(pars[1]) # nb. combinations in pars
    verbose ? println("-- Nb. combinations = ", ncomb) : nothing
    res = map(values(pars)...) do v...
        verbose ? println(Pair.(keys(pars), v)...) : nothing
        fm = fun(Xtrain, Ytrain; Pair.(keys(pars), v)...)
        pred = Jchemo.predict(fm, X).pred
        score(pred, Y)
    end
    verbose ? println("-- End.") : nothing
    ncomb == 1 ? res = res[1] : res = reduce(vcat, res) 
    dat = DataFrame(pars)
    namy = map(string, repeat(["y"], q), 1:q)
    res = DataFrame(res, Symbol.(namy))
    hcat(dat, res)
end


